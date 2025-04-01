from flask import Flask, request, jsonify
import os
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
import uuid
import json
from datetime import datetime
import time
import threading
import queue
from bs4 import BeautifulSoup

# Qdrant client
import qdrant_client
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

# LlamaIndex imports
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorService")

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re


class SemanticChunker:
    """高级中文语义分块器，使用深度学习模型自适应识别语义边界"""

    def __init__(
            self,
            model_path="/models/chinese-roberta-wwm-ext",
            base_threshold=0.65,
            min_chunk_chars=150,  # 确保短文本至少作为一个块
            max_chunk_chars=1500,  # 最大分块尺寸
            device="cpu",
            batch_size=8,  # 批处理大小
            adaptive_threshold=True,  # 自适应阈值
            debug_mode=False
    ):
        """
        初始化高级语义分块器

        Args:
            model_path: 语义模型路径
            base_threshold: 基础相似度阈值 (0-1)，较低的值会产生更多的分块
            min_chunk_chars: 最小块字符数
            max_chunk_chars: 最大块字符数
            device: 计算设备 ('cpu' 或 'cuda:0' 等)
            batch_size: 嵌入计算的批处理大小
            adaptive_threshold: 是否使用自适应阈值
            debug_mode: 是否启用调试模式
        """
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.base_threshold = base_threshold
        self.adaptive_threshold = adaptive_threshold
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        self.logger = logging.getLogger("SemanticChunker")

        # 加载分词器和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.device = device
            self.model.to(device)
            self.model.eval()  # 设置为评估模式

            # 中文特定分隔符模式
            self.cn_sentence_pattern = re.compile(r'([。！？\!\?]+|[\n]{2,})')
            self.cn_subsentence_pattern = re.compile(r'([，；：、,;])')

            # 缓存最近处理的文档向量，提高连续处理性能
            self.embedding_cache = {}
            self.cache_size = 100  # 最多缓存100个句子的嵌入

            self.logger.info(f"语义分块器初始化成功: 模型={model_path}, 设备={device}")
        except Exception as e:
            self.logger.error(f"初始化语义分块器失败: {str(e)}")
            raise RuntimeError(f"加载语义模型失败: {str(e)}")

    def _get_sentence_embedding(self, sentences):
        """批量获取句子嵌入向量"""
        if not sentences:
            return []

        # 批处理嵌入计算，降低内存占用
        embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]

            # 检查缓存
            batch_embeddings = []
            new_sentences = []
            new_indices = []

            for j, sentence in enumerate(batch):
                if sentence in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[sentence])
                else:
                    new_sentences.append(sentence)
                    new_indices.append(j)

            # 处理未缓存的句子
            if new_sentences:
                # 确保输入有效
                valid_sentences = [s if s.strip() else "空" for s in new_sentences]

                # 使用模型获取嵌入
                with torch.no_grad():
                    inputs = self.tokenizer(valid_sentences, padding=True, truncation=True,
                                            return_tensors="pt", max_length=256).to(self.device)
                    outputs = self.model(**inputs)

                    # 使用CLS嵌入作为句子表示
                    new_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # 更新缓存
                for k, sentence in enumerate(new_sentences):
                    embedding = new_embeddings[k]
                    self.embedding_cache[sentence] = embedding

                    # 在正确位置插入新嵌入
                    if k < len(new_indices):
                        batch_embeddings.insert(new_indices[k], embedding)

                # 限制缓存大小
                if len(self.embedding_cache) > self.cache_size:
                    # 删除最早添加的项
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]

            embeddings.extend(batch_embeddings)

        return embeddings

    def _calculate_similarity(self, emb1, emb2):
        """计算两个嵌入向量的余弦相似度"""
        # 长度归一化，避免长度偏差
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0:
            return np.dot(emb1, emb2) / (norm1 * norm2)
        return 0.0

    def _compute_adaptive_threshold(self, similarities):
        """动态计算最优阈值"""
        if not similarities or len(similarities) < 3:
            return self.base_threshold

        # 使用四分位范围法检测异常点
        q1 = np.percentile(similarities, 25)
        q3 = np.percentile(similarities, 75)
        iqr = q3 - q1

        # 调整阈值以捕获显著的语义变化
        lower_bound = q1 - 1.5 * iqr
        threshold = max(lower_bound, self.base_threshold * 0.8)
        return min(threshold, self.base_threshold * 1.2)  # 限制范围

    def chunk_text(self, text, min_chunk_size=None, max_chunk_size=None):
        """
        核心方法：将文本分割成语义连贯的块

        Args:
            text: 要分割的文本
            min_chunk_size: 可选，最小块大小，覆盖初始化值
            max_chunk_size: 可选，最大块大小，覆盖初始化值

        Returns:
            List[str]: 分割后的文本块列表
        """
        if not text:
            return []

        # 使用传入的参数或默认值
        min_size = min_chunk_size if min_chunk_size is not None else self.min_chunk_chars
        max_size = max_chunk_size if max_chunk_size is not None else self.max_chunk_chars

        # 短文本直接作为一个块返回
        if len(text) < min_size:
            return [text]

        try:
            # 阶段1: 按段落进行初步分割
            paragraphs = self._split_into_paragraphs(text)

            # 阶段2: 对每个段落进行语义边界检测
            chunks = []
            current_chunk = ""

            for para in paragraphs:
                # 如果段落很短，直接添加到当前块
                if len(para) < min_size // 3:  # 非常短的段落
                    if current_chunk and len(current_chunk) + len(para) + 2 <= max_size:
                        current_chunk += "\n\n" + para
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = para
                    continue

                # 对较长段落进行句子级分析
                para_chunks = self._find_semantic_boundaries_in_paragraph(para)

                for chunk in para_chunks:
                    if not current_chunk:
                        current_chunk = chunk
                    elif len(current_chunk) + len(chunk) + 2 <= max_size:
                        current_chunk += "\n\n" + chunk
                    else:
                        chunks.append(current_chunk)
                        current_chunk = chunk

            # 添加最后一个块
            if current_chunk:
                chunks.append(current_chunk)

            # 阶段3: 处理过大的块
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= max_size:
                    final_chunks.append(chunk)
                else:
                    # 对过大块进行后处理拆分
                    sub_chunks = self._split_oversized_chunk(chunk, max_size)
                    final_chunks.extend(sub_chunks)

            # 确保至少返回一个块
            if not final_chunks:
                return [text]

            if self.debug_mode:
                self.logger.info(f"原始长度: {len(text)}, 分块数: {len(final_chunks)}")
                for i, chunk in enumerate(final_chunks):
                    self.logger.info(f"块 {i + 1}/{len(final_chunks)}: {len(chunk)} 字符")

            return final_chunks

        except Exception as e:
            self.logger.error(f"分块过程中出错: {str(e)}")
            # 出错时，确保至少返回原始文本作为一个块
            return [text]

    def _split_into_paragraphs(self, text):
        """将文本拆分为段落"""
        # 按照多种可能的段落分隔符拆分
        raw_paragraphs = re.split(r'(\n\s*\n|\r\n\s*\r\n)', text)

        # 重新组合段落和分隔符
        paragraphs = []
        for i in range(0, len(raw_paragraphs), 2):
            para = raw_paragraphs[i]
            if para.strip():
                paragraphs.append(para.strip())

        # 如果没有检测到段落，将整个文本作为一个段落
        if not paragraphs:
            paragraphs = [text]

        return paragraphs

    def _find_semantic_boundaries_in_paragraph(self, paragraph):
        """在段落中检测语义边界"""
        # 分割成句子 (中文特定)
        sentences = []
        parts = self.cn_sentence_pattern.split(paragraph)

        for i in range(0, len(parts) - 1, 2):
            if i < len(parts):
                sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
                if sentence.strip():
                    sentences.append(sentence.strip())

        # 如果只有一个或没有句子，返回整个段落
        if len(sentences) <= 1:
            return [paragraph]

        # 获取句子嵌入
        embeddings = self._get_sentence_embedding(sentences)

        if len(embeddings) <= 1:
            return [paragraph]

        # 计算相邻句子的相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._calculate_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 计算自适应阈值
        threshold = self._compute_adaptive_threshold(similarities) if self.adaptive_threshold else self.base_threshold

        # 识别边界
        boundaries = [0]  # 起始点
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)

        # 添加结束边界
        boundaries.append(len(sentences))

        # 根据边界生成块
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk = " ".join(sentences[start:end])
            chunks.append(chunk)

        return chunks

    def _split_oversized_chunk(self, chunk, max_size):
        """处理超过最大大小的块"""
        # 首先尝试在句子边界分割
        sentences = self.cn_sentence_pattern.split(chunk)
        if not sentences:
            return [chunk]  # 无法分割

        # 重组句子并检查大小
        sub_chunks = []
        current = ""
        for i in range(0, len(sentences) - 1, 2):
            if i < len(sentences):
                sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")

                if not current:
                    current = sentence
                elif len(current) + len(sentence) <= max_size:
                    current += sentence
                else:
                    sub_chunks.append(current)
                    current = sentence

        if current:
            sub_chunks.append(current)

        # 如果存在极长句子，可能需要进一步分割
        final_sub_chunks = []
        for sub in sub_chunks:
            if len(sub) <= max_size:
                final_sub_chunks.append(sub)
            else:
                # 以子句分隔符进一步分割
                parts = self.cn_subsentence_pattern.split(sub)
                current = ""
                for i in range(0, len(parts) - 1, 2):
                    if i < len(parts):
                        part = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")

                        if not current:
                            current = part
                        elif len(current) + len(part) <= max_size:
                            current += part
                        else:
                            final_sub_chunks.append(current)
                            current = part

                if current:
                    final_sub_chunks.append(current)

        # 如果所有方法都失败，强制截断
        if not final_sub_chunks:
            # 每max_size字符强制截断
            return [chunk[i:i + max_size] for i in range(0, len(chunk), max_size)]

        return final_sub_chunks
# 创建自己的嵌入适配器
class CustomEmbedding:
    def __init__(self, model_path, local_files_only=True):
        """初始化自定义嵌入模型

        Args:
            model_path: 模型路径
            local_files_only: 是否仅使用本地模型
        """
        self.model = SentenceTransformer(model_path, local_files_only=local_files_only)
        self.tokenizer = self.model.tokenizer
        # 设置模型属性，LlamaIndex可能会查询这些
        self.model_name = model_path
        self.embed_dim = self.model.get_sentence_embedding_dimension()

    def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取多个文本的嵌入向量"""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """批量获取文本嵌入向量 - LlamaIndex内部调用此方法

        Args:
            texts: 文本列表
            **kwargs: 额外参数，包括show_progress等

        Returns:
            嵌入向量列表
        """
        # 忽略额外参数，只传递SentenceTransformer支持的参数
        show_progress = kwargs.get('show_progress', False)
        # sentence-transformers支持show_progress
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=show_progress).tolist()

    def embed_documents(self, documents: List[Any], **kwargs) -> List[List[float]]:
        """将文档对象转换为嵌入向量列表"""
        if hasattr(documents[0], "text"):
            # 如果是文档对象有text属性
            texts = [doc.text for doc in documents]
        else:
            # 假设文档对象可以转换为字符串
            texts = [str(doc) for doc in documents]
        return self.get_text_embedding_batch(texts, **kwargs)

    def embed_query(self, query: str) -> List[float]:
        """将查询转换为嵌入向量"""
        return self.get_text_embedding(query)

    def to_dict(self) -> Dict:
        """序列化方法，LlamaIndex有时会调用此方法保存模型配置"""
        return {
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
            "type": "custom_embedding"
        }

    def get_agg_embedding_from_queries(self, queries: List[str], **kwargs) -> List[float]:
        """优化的聚合嵌入方法（归一化后平均）"""
        embeddings = [self.get_text_embedding(q) for q in queries]

        # 先对每个向量进行L2归一化
        normalized = []
        for emb in embeddings:
            norm = sum(x * x for x in emb) ** 0.5
            if norm > 0:
                normalized.append([x / norm for x in emb])
            else:
                normalized.append(emb)

        # 然后平均
        embedding_length = len(normalized[0])
        agg_embedding = [0.0] * embedding_length
        for emb in normalized:
            for i in range(embedding_length):
                agg_embedding[i] += emb[i]

        for i in range(embedding_length):
            agg_embedding[i] /= len(normalized)

        # 最后再次归一化
        norm = sum(x * x for x in agg_embedding) ** 0.5
        if norm > 0:
            return [x / norm for x in agg_embedding]
        return agg_embedding

class AsyncTaskManager:
    """异步任务管理器，使用线程池处理耗时任务"""

    def __init__(self, max_workers=5):
        """
        初始化任务管理器

        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.workers = []
        self.results = {}  # 存储任务结果
        self.status = {}  # 存储任务状态 (pending, running, completed, failed)
        self.callbacks = {}  # 任务完成后的回调函数
        self.logger = logging.getLogger("AsyncTaskManager")
        self._start_workers()

    def _worker_loop(self):
        """工作线程循环，不断从队列获取任务并执行"""
        while True:
            try:
                # 从队列获取任务
                task_id, task_func, args, kwargs = self.task_queue.get()

                # 更新任务状态
                self.status[task_id] = "running"
                self.logger.info(f"开始执行任务 {task_id}")

                try:
                    # 执行任务
                    result = task_func(*args, **kwargs)
                    # 存储结果
                    self.results[task_id] = result
                    self.status[task_id] = "completed"
                    self.logger.info(f"任务 {task_id} 完成")

                    # 执行回调（如果有）
                    if task_id in self.callbacks and self.callbacks[task_id]:
                        try:
                            self.callbacks[task_id](result)
                            self.logger.info(f"任务 {task_id} 回调执行成功")
                        except Exception as e:
                            self.logger.error(f"任务 {task_id} 回调执行失败: {str(e)}")

                except Exception as e:
                    # 任务执行失败
                    self.status[task_id] = "failed"
                    self.results[task_id] = str(e)
                    self.logger.error(f"任务 {task_id} 执行失败: {str(e)}")

                # 标记任务完成
                self.task_queue.task_done()

            except Exception as e:
                self.logger.error(f"工作线程执行出错: {str(e)}")
                # 短暂休息以避免CPU占用过高
                time.sleep(0.1)

    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.workers.append(thread)
            self.logger.info(f"启动工作线程 {i + 1}")

    def submit_task(self, task_func, callback=None, *args, **kwargs) -> str:
        """
        提交任务到队列

        Args:
            task_func: 要执行的函数
            callback: 任务完成后的回调函数，接收任务结果作为参数
            *args, **kwargs: 传递给任务函数的参数

        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        self.status[task_id] = "pending"

        if callback:
            self.callbacks[task_id] = callback

        # 将任务放入队列
        self.task_queue.put((task_id, task_func, args, kwargs))
        self.logger.info(f"提交任务 {task_id} 到队列")

        return task_id

    def get_task_status(self, task_id: str) -> Dict:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict: 包含任务状态和结果（如果已完成）
        """
        if task_id not in self.status:
            return {"status": "not_found"}

        result = {
            "status": self.status[task_id]
        }

        # 如果任务已完成或失败，包含结果
        if self.status[task_id] in ["completed", "failed"] and task_id in self.results:
            result["result"] = self.results[task_id]

        return result

class QdrantLlamaIndexService:
    """使用Qdrant和LlamaIndex的向量存储和检索服务"""

    def __init__(
            self,
            embedding_model_path: str = "/models/sentence-transformers_text2vec-large-chinese",
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            similarity_top_k: int = 5,
            use_semantic_chunking: bool = True  # 控制是否使用语义分块
    ):
        """
        初始化向量服务

        Args:
            embedding_model_path: 嵌入模型路径
            qdrant_host: Qdrant服务器地址
            qdrant_port: Qdrant服务器端口
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            similarity_top_k: 默认搜索返回结果数
        """
        self.logger = logging.getLogger("QdrantLlamaIndexService")


        # 文本分块配置
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        # 语义分块设置
        self.use_semantic_chunking = use_semantic_chunking
        self.semantic_chunker = None

        if self.use_semantic_chunking:
            try:
                self.semantic_chunker = SemanticChunker(
                    model_path="/models/chinese-roberta-wwm-ext",
                    base_threshold=0.65,  # 可调整的基础阈值
                    min_chunk_chars=150,  # 确保短文本也能被处理
                    max_chunk_chars=1500,  # 最大块大小
                    adaptive_threshold=True
                )
                logger.info("语义分块模型加载成功")
            except Exception as e:
                logger.error(f"加载语义分块模型失败: {str(e)}")
                self.use_semantic_chunking = False
                logger.info("已回退到标准分块方法")


        # 初始化Qdrant客户端
        self.logger.info(f"连接到Qdrant: {qdrant_host}:{qdrant_port}")
        self.qdrant_client = qdrant_client.QdrantClient(
            host=qdrant_host,
            port=qdrant_port
        )

        # 确保集合存在
        self._create_collections()

        # 设置LlamaIndex
        self.logger.info(f"加载嵌入模型: {embedding_model_path}")
        try:
            # 使用LlamaIndex自带的适配器，添加本地模型参数

            self.embed_model = CustomEmbedding(embedding_model_path)

            # 设置LlamaIndex全局配置
            Settings.embed_model = self.embed_model
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap

            # 初始化文本分割器
            self.splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator="\n\n",  # 使用双换行符作为段落分隔符
                secondary_chunking_regex="。|？|！|；|;|\\?|!|\\.",  # 中文句子分隔符
                tokenizer=self.embed_model.tokenizer
            )

            # 创建向量索引
            self.news_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name="cooper_news"
            )

            self.notice_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name="cooper_notice"
            )

            self.news_index = VectorStoreIndex.from_vector_store(self.news_vector_store)
            self.notice_index = VectorStoreIndex.from_vector_store(self.notice_vector_store)

            self.logger.info("向量服务初始化完成")
        except Exception as e:
            self.logger.error(f"加载嵌入模型时出错: {str(e)}")
            raise

    def _create_collections(self):
        """确保Qdrant集合存在"""
        # 获取现有集合列表
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        # 检查并创建新闻集合
        if "cooper_news" not in collection_names:
            self.logger.info("创建cooper_news集合")
            self.qdrant_client.create_collection(
                collection_name="cooper_news",
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_dim if hasattr(self, 'vector_dim') else 1024,  # 使用默认维度或已知维度
                    distance=qdrant_models.Distance.COSINE
                )
            )

        # 检查并创建公告集合
        if "cooper_notice" not in collection_names:
            self.logger.info("创建cooper_notice集合")
            self.qdrant_client.create_collection(
                collection_name="cooper_notice",
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_dim if hasattr(self, 'vector_dim') else 1024,  # 使用默认维度或已知维度
                    distance=qdrant_models.Distance.COSINE
                )
            )

    def add_news_async(self,
                       title: str,
                       content: str,
                       source: str = None,
                       publish_date: str = None,
                       tags: List[str] = None,
                       id: str = None) -> str:
        """
        异步添加新闻

        Args:
            title: 新闻标题
            content: 新闻内容
            source: 新闻来源
            publish_date: 发布日期
            tags: 标签列表
            id: 文档ID

        Returns:
            str: 任务ID
        """
        """异步添加新闻"""
        # 生成基础ID
        base_id = id or str(uuid.uuid4())

        # 准备要在线程中执行的任务函数
        def process_and_add_task():
            try:
                # 准备元数据
                metadata = {
                    "title": title,
                    "source": source or "未知来源",
                    "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                    "type": "news",
                    "base_id": base_id
                }

                if tags:
                    if isinstance(tags, list):
                        metadata["tags"] = ",".join(tags)
                    else:
                        metadata["tags"] = tags

                # 使用语义分块或标准分块
                if self.use_semantic_chunking and self.semantic_chunker:
                    logger.info(f"使用语义分块处理新闻: {title}")

                    # 使用语义分块器进行分块
                    full_text = f"{title}\n{content}"
                    text_chunks = self.semantic_chunker.chunk_text(
                        full_text,
                        min_chunk_size=self.chunk_size // 2,
                        max_chunk_size=self.chunk_size
                    )

                    # 为每个语义块创建Document（非Node！）
                    documents = []
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = i
                        chunk_metadata["total_chunks"] = len(text_chunks)

                        # 增强内容 - 为每个块添加标题提示
                        if not chunk_text.startswith(title):
                            chunk_text = f"{title} - 片段{i + 1}/{len(text_chunks)}\n\n{chunk_text}"

                        # 创建Document对象，非Node对象
                        doc = Document(
                            text=chunk_text,
                            metadata=chunk_metadata
                        )
                        documents.append(doc)

                    # 直接从多个文档创建节点
                    nodes = self.splitter.get_nodes_from_documents(documents)

                    # 为节点设置ID
                    for i, node in enumerate(nodes):
                        node.id_ = str(uuid.uuid4())  # 全新UUID
                        node.metadata["base_id"] = base_id  # 必须保存原始ID
                    logger.info(f"使用语义分块将新闻拆分为 {len(nodes)} 个节点")
                else:
                    # 使用标准LlamaIndex分块
                    logger.info(f"使用标准分块处理新闻: {title}")
                    documents = [Document(
                        text=f"{title}\n{content}",
                        metadata=metadata
                    )]

                    # 使用原有分块器
                    nodes = self.splitter.get_nodes_from_documents(documents)

                    # 为每个节点设置ID
                    for i, node in enumerate(nodes):
                        node.id_ = str(uuid.uuid4())  # 全新UUID
                        node.metadata["base_id"] = base_id  # 必须保存原始ID
                        node.metadata["chunk_index"] = i
                        node.metadata["total_chunks"] = len(nodes)

                        # 增强内容 - 为每个块添加标题提示
                        if not node.text.startswith(title):
                            node.text = f"{title} - 片段{i + 1}/{len(nodes)}\n\n{node.text}"

                    logger.info(f"使用标准分块将新闻拆分为 {len(nodes)} 个节点")

                # 添加到向量索引
                self.news_index.insert_nodes(nodes)

                return {
                    "id": base_id,
                    "chunks": len(nodes),
                    "title": title,
                    "source": source or "未知来源",
                    "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                    "chunking_method": "semantic" if self.use_semantic_chunking and self.semantic_chunker else "standard"
                }

            except Exception as e:
                logger.error(f"异步添加新闻时出错: {str(e)}")
                raise
        # 提交到任务管理器
        task_id = task_manager.submit_task(process_and_add_task)
        self.logger.info(f"提交添加新闻任务: {task_id}, 文档基础ID: {base_id}")

        return task_id
    def add_announcement_async(self,
                               title: str,
                               content: str,
                               department: str = None,
                               publish_date: str = None,
                               importance: str = "normal",
                               id: str = None) -> str:
        """
        异步添加公告，支持语义分块

        Args:
            title: 公告标题
            content: 公告内容
            department: 发布部门
            publish_date: 发布日期
            importance: 重要性
            id: 文档ID

        Returns:
            str: 任务ID
        """
        # 生成基础ID
        base_id = id or str(uuid.uuid4())

        # 准备要在线程中执行的任务函数
        def process_and_add_task():
            try:
                # 准备元数据
                metadata = {
                    "title": title,
                    "department": department or "未知部门",
                    "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                    "importance": importance,
                    "type": "announcement",
                    "base_id": base_id
                }

                # 判断是否使用语义分块
                if self.use_semantic_chunking and self.semantic_chunker:
                    self.logger.info(f"使用语义分块处理公告: {title}")

                    # 使用语义分块器对完整文本进行分块
                    full_text = f"{title}\n{content}"
                    text_chunks = self.semantic_chunker.chunk_text(
                        full_text,
                        min_chunk_size=self.chunk_size // 2,
                        max_chunk_size=self.chunk_size
                    )

                    # 为每个语义块创建Document对象
                    documents = []
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = i
                        chunk_metadata["total_chunks"] = len(text_chunks)

                        # 增强内容 - 为每个块添加标题和重要性标记
                        if not chunk_text.startswith(title):
                            chunk_text = f"{title} ({importance}) - 片段{i + 1}/{len(text_chunks)}\n\n{chunk_text}"

                        # 创建Document对象
                        doc = Document(
                            text=chunk_text,
                            metadata=chunk_metadata
                        )
                        documents.append(doc)

                    # 从文档创建节点 - 使用现有分块器
                    nodes = self.splitter.get_nodes_from_documents(documents)

                    # 为节点设置ID
                    for i, node in enumerate(nodes):
                        node.id_ = str(uuid.uuid4())  # 全新UUID
                        node.metadata["base_id"] = base_id  # 必须保存原始ID

                    self.logger.info(f"使用语义分块将公告拆分为 {len(nodes)} 个块")
                else:
                    # 使用标准分块流程
                    self.logger.info(f"使用标准分块处理公告: {title}")

                    # 创建LlamaIndex文档
                    documents = [Document(
                        text=f"{title}\n{content}",
                        metadata=metadata
                    )]

                    # 分割文档
                    nodes = self.splitter.get_nodes_from_documents(documents)

                    # 为每个节点设置文档ID和其他元数据
                    for i, node in enumerate(nodes):
                        node.id_ = str(uuid.uuid4())  # 全新UUID
                        node.metadata["base_id"] = base_id  # 必须保存原始ID
                        node.metadata["chunk_index"] = i
                        node.metadata["total_chunks"] = len(nodes)

                        # 增强内容 - 为每个块添加标题提示
                        if not node.text.startswith(title):
                            node.text = f"{title} ({importance}) - 片段{i + 1}/{len(nodes)}\n\n{node.text}"

                    self.logger.info(f"使用标准分块将公告拆分为 {len(nodes)} 个块")

                # 添加到向量索引
                self.notice_index.insert_nodes(nodes)
                self.logger.info(f"成功添加公告: {title}，分为 {len(nodes)} 个块")

                return {
                    "id": base_id,
                    "chunks": len(nodes),
                    "title": title,
                    "department": department or "未知部门",
                    "importance": importance,
                    "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                    "chunking_method": "semantic" if self.use_semantic_chunking and self.semantic_chunker else "standard"
                }

            except Exception as e:
                self.logger.error(f"异步添加公告时出错: {str(e)}")
                raise

        # 提交到任务管理器
        task_id = task_manager.submit_task(process_and_add_task)
        self.logger.info(f"提交添加公告任务: {task_id}, 文档基础ID: {base_id}")

        return task_id

    def search_news(self, query: str, n_results: int = None) -> List[Dict]:
        """
        搜索新闻

        Args:
            query: 查询文本
            n_results: 返回结果数量，如果为None则使用默认值

        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 使用指定的参数或默认参数
            top_k = n_results if n_results is not None else self.similarity_top_k

            # 创建检索器
            retriever = VectorIndexRetriever(
                index=self.news_index,
                similarity_top_k=top_k * 2,  # 获取更多结果用于去重
            )


            nodes = retriever.retrieve(query)

            # 格式化结果
            formatted_results = []
            seen_base_ids = set()

            for node in nodes:
                base_id = node.metadata.get("base_id", node.id_)

                # 如果已经包含了这个base_id的文档，则跳过
                if base_id in seen_base_ids:
                    continue

                seen_base_ids.add(base_id)

                formatted_results.append({
                    "id": base_id,
                    "title": node.metadata.get("title", ""),
                    "source": node.metadata.get("source", "未知来源"),
                    "publish_date": node.metadata.get("publish_date", ""),
                    "content": node.text,
                    "relevance_score": node.score if hasattr(node, "score") else 0.0,
                    "search_type": "vector"
                })

                # 如果已经收集了足够的不同文档，就停止
                if len(formatted_results) >= top_k:
                    break

            return formatted_results
        except Exception as e:
            self.logger.error(f"搜索新闻时出错: {str(e)}")
            return []

    def search_announcements(self, query: str, n_results: int = None) -> List[Dict]:
        """
        搜索公告

        Args:
            query: 查询文本
            n_results: 返回结果数量，如果为None则使用默认值

        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 使用指定的参数或默认参数
            top_k = n_results if n_results is not None else self.similarity_top_k


            # 创建检索器
            retriever = VectorIndexRetriever(
                index=self.notice_index,
                similarity_top_k=top_k * 2,  # 获取更多结果用于去重
            )

            # 执行检索
            nodes = retriever.retrieve(query)

            # 格式化结果
            formatted_results = []
            seen_base_ids = set()

            for node in nodes:
                base_id = node.metadata.get("base_id", node.id_)

                # 如果已经包含了这个base_id的文档，则跳过
                if base_id in seen_base_ids:
                    continue

                seen_base_ids.add(base_id)

                formatted_results.append({
                    "id": base_id,
                    "title": node.metadata.get("title", ""),
                    "department": node.metadata.get("department", "未知部门"),
                    "publish_date": node.metadata.get("publish_date", ""),
                    "importance": node.metadata.get("importance", "normal"),
                    "content": node.text,
                    "relevance_score": node.score if hasattr(node, "score") else 0.0,
                    "search_type": "vector"
                })

                # 如果已经收集了足够的不同文档，就停止
                if len(formatted_results) >= top_k:
                    break

            return formatted_results
        except Exception as e:
            self.logger.error(f"搜索公告时出错: {str(e)}")
            return []

    def search_all(self, query: str, n_results: int = None) -> Dict[str, List[Dict]]:
        """
        同时搜索新闻和公告

        Args:
            query: 查询文本
            n_results: 每种类型返回的结果数量

        Returns:
            Dict[str, List[Dict]]: 搜索结果字典
        """
        news_results = self.search_news(query, n_results)
        announcement_results = self.search_announcements(query, n_results)

        return {
            "news": news_results,
            "announcements": announcement_results
        }

    def delete_news(self, doc_id: str) -> bool:
        """
        删除新闻

        Args:
            doc_id: 文档ID

        Returns:
            bool: 是否成功删除
        """
        try:
            # 检查ID是否包含块索引
            if '_' in doc_id:
                base_id = doc_id.split('_')[0]
                # 查找并删除所有相关块
                results = self.qdrant_client.scroll(
                    collection_name="cooper_news",
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.base_id",
                                match=qdrant_models.MatchValue(value=base_id)
                            )
                        ]
                    ),
                    limit=100
                )

                if results and results[0]:
                    # 获取所有ID
                    ids = [point.id for point in results[0]]
                    # 删除所有匹配的文档
                    self.qdrant_client.delete(
                        collection_name="cooper_news",
                        points_selector=qdrant_models.PointIdsList(
                            points=ids
                        )
                    )
                    self.logger.info(f"成功删除新闻ID: {base_id} 的所有 {len(ids)} 个块")
                    return True
                else:
                    return False
            else:
                # 尝试按base_id删除
                results = self.qdrant_client.scroll(
                    collection_name="cooper_news",
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.base_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    ),
                    limit=100
                )

                if results and results[0]:
                    # 获取所有ID
                    ids = [point.id for point in results[0]]
                    # 删除所有匹配的文档
                    self.qdrant_client.delete(
                        collection_name="cooper_news",
                        points_selector=qdrant_models.PointIdsList(
                            points=ids
                        )
                    )
                    self.logger.info(f"成功删除新闻ID: {doc_id} 的所有 {len(ids)} 个块")
                    return True
                else:
                    # 尝试直接删除该ID
                    self.qdrant_client.delete(
                        collection_name="cooper_news",
                        points_selector=qdrant_models.PointIdsList(
                            points=[doc_id]
                        )
                    )
                    self.logger.info(f"成功删除新闻ID: {doc_id}")
                    return True
        except Exception as e:
            self.logger.error(f"删除新闻时出错: {str(e)}")
            return False

    def delete_notice(self, doc_id: str) -> bool:
        """
        删除公告

        Args:
            doc_id: 文档ID

        Returns:
            bool: 是否成功删除
        """
        try:
            # 检查ID是否包含块索引
            if '_' in doc_id:
                base_id = doc_id.split('_')[0]
                # 查找并删除所有相关块
                results = self.qdrant_client.scroll(
                    collection_name="cooper_notice",
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.base_id",
                                match=qdrant_models.MatchValue(value=base_id)
                            )
                        ]
                    ),
                    limit=100
                )

                if results and results[0]:
                    # 获取所有ID
                    ids = [point.id for point in results[0]]
                    # 删除所有匹配的文档
                    self.qdrant_client.delete(
                        collection_name="cooper_notice",
                        points_selector=qdrant_models.PointIdsList(
                            points=ids
                        )
                    )
                    self.logger.info(f"成功删除公告ID: {base_id} 的所有 {len(ids)} 个块")
                    return True
                else:
                    return False
            else:
                # 尝试按base_id删除
                results = self.qdrant_client.scroll(
                    collection_name="cooper_notice",
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.base_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    ),
                    limit=100
                )

                if results and results[0]:
                    # 获取所有ID
                    ids = [point.id for point in results[0]]
                    # 删除所有匹配的文档
                    self.qdrant_client.delete(
                        collection_name="cooper_notice",
                        points_selector=qdrant_models.PointIdsList(
                            points=ids
                        )
                    )
                    self.logger.info(f"成功删除公告ID: {doc_id} 的所有 {len(ids)} 个块")
                    return True
                else:
                    # 尝试直接删除该ID
                    self.qdrant_client.delete(
                        collection_name="cooper_notice",
                        points_selector=qdrant_models.PointIdsList(
                            points=[doc_id]
                        )
                    )
                    self.logger.info(f"成功删除公告ID: {doc_id}")
                    return True
        except Exception as e:
            self.logger.error(f"删除公告时出错: {str(e)}")
            return False


# 创建任务管理器
task_manager = AsyncTaskManager(max_workers=3)

# 从环境变量获取配置
EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', '/models/sentence-transformers_text2vec-large-chinese')
QDRANT_HOST = os.environ.get('QDRANT_HOST', '124.71.225.73')
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', '6333'))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))
SIMILARITY_TOP_K = int(os.environ.get('SIMILARITY_TOP_K', '5'))
MMR_DIVERSITY_BIAS = float(os.environ.get('MMR_DIVERSITY_BIAS', '0.3'))

# 应用启动时初始化向量服务
vector_service = None

with app.app_context():
    try:
        logger.info("初始化向量服务...")
        vector_service = QdrantLlamaIndexService(
            embedding_model_path=EMBEDDING_MODEL_PATH,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            similarity_top_k=SIMILARITY_TOP_K,
        )
        logger.info("向量服务初始化完成")
    except Exception as e:
        logger.error(f"初始化向量服务时出错: {str(e)}")
        raise


# 路由定义
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    global vector_service
    status = "healthy" if vector_service is not None else "unhealthy"
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "qdrant_host": QDRANT_HOST,
            "qdrant_port": QDRANT_PORT,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    })


@app.route('/vector/search', methods=['POST'])
def vector_search():
    """向量搜索接口"""
    global vector_service
    if not vector_service:
        return jsonify({"error": "向量服务尚未初始化"}), 500

    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "请提供查询内容"}), 400

    query = data['query']
    n_results = data.get('n_results', None)
    search_type = data.get('search_type', 'all').lower()

    try:
        if search_type == 'news':
            results = {"news": vector_service.search_news(query, n_results), "announcements": []}
        elif search_type == 'announcements':
            results = {"news": [], "announcements": vector_service.search_announcements(query, n_results)}
        else:  # 'all' or any other value
            results = vector_service.search_all(query, n_results)

        return app.response_class(
            response=json.dumps(results, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"向量搜索出错: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/news', methods=['POST'])
def add_news():
    """添加新闻接口"""
    global vector_service
    if not vector_service:
        return jsonify({"error": "向量服务尚未初始化"}), 500

    data = request.json
    if not data or 'title' not in data or 'content' not in data:
        return jsonify({"error": "请提供标题和内容"}), 400

    # 获取参数
    title = data['title']
    content = data['content']
    source = data.get('source')
    publish_date = data.get('publish_date')
    tags = data.get('tags')
    doc_id = data.get('id')

    try:
        # 提交异步任务
        task_id = vector_service.add_news_async(
            title=title,
            content=content,
            source=source,
            publish_date=publish_date,
            tags=tags,
            id=doc_id
        )

        return jsonify({
            "success": True,
            "message": "新闻添加任务已提交",
            "task_id": task_id
        })
    except Exception as e:
        logger.error(f"添加新闻出错: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/announcement', methods=['POST'])
def add_announcement():
    """添加公告接口"""
    global vector_service
    if not vector_service:
        return jsonify({"error": "向量服务尚未初始化"}), 500

    data = request.json
    if not data or 'title' not in data or 'content' not in data:
        return jsonify({"error": "请提供标题和内容"}), 400

    # 获取参数
    title = data['title']
    content = data['content']
    department = data.get('department')
    publish_date = data.get('publish_date')
    importance = data.get('importance', 'normal')
    doc_id = data.get('id')

    try:
        # 提交异步任务
        task_id = vector_service.add_announcement_async(
            title=title,
            content=content,
            department=department,
            publish_date=publish_date,
            importance=importance,
            id=doc_id
        )

        return jsonify({
            "success": True,
            "message": "公告添加任务已提交",
            "task_id": task_id
        })
    except Exception as e:
        logger.error(f"添加公告出错: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态接口"""
    status = task_manager.get_task_status(task_id)
    return jsonify(status)


@app.route('/news/<doc_id>', methods=['DELETE'])
def delete_news(doc_id):
    """删除新闻接口"""
    global vector_service
    if not vector_service:
        return jsonify({"error": "向量服务尚未初始化"}), 500

    try:
        success = vector_service.delete_news(doc_id)
        if success:
            return jsonify({
                "success": True,
                "message": f"成功删除新闻: {doc_id}"
            })
        else:
            return jsonify({"error": f"找不到ID为{doc_id}的新闻"}), 404
    except Exception as e:
        logger.error(f"删除新闻出错: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/announcement/<doc_id>', methods=['DELETE'])
def delete_announcement(doc_id):
    """删除公告接口"""
    global vector_service
    if not vector_service:
        return jsonify({"error": "向量服务尚未初始化"}), 500

    try:
        success = vector_service.delete_notice(doc_id)
        if success:
            return jsonify({
                "success": True,
                "message": f"成功删除公告: {doc_id}"
            })
        else:
            return jsonify({"error": f"找不到ID为{doc_id}的公告"}), 404
    except Exception as e:
        logger.error(f"删除公告出错: {str(e)}")
        return jsonify({"error": str(e)}), 500


# 启动应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)