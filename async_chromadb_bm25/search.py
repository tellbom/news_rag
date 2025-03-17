from flask import Flask, request, jsonify
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import uuid
from typing import List, Dict, Optional, Union, Any, Tuple
import json
from datetime import datetime
import os
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import requests
from paddleocr import PaddleOCR
import mammoth
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
import tempfile
import threading
import queue
import time
import uuid
from typing import Dict, Callable, Any, List, Optional
import logging

from flask import Flask
app = Flask(__name__)

class ESServiceClient:
    """
    ES服务客户端，用于与ES服务API通信
    """
    
    def __init__(self, base_url: str = "http://localhost:8085"):
        """
        初始化ES服务客户端
        
        Args:
            base_url: ES服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.is_available = self._check_health()
        self.session = requests.Session()
    
    def _check_health(self) -> bool:
        """
        检查ES服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception as e:
            app.logger.error(f"ES服务健康检查失败: {str(e)}")
            return False
    
    def index_news(self, document: Dict, async_mode: bool = True) -> bool:
        """
        索引新闻到ES
        
        Args:
            document: 新闻文档
            async_mode: 是否异步执行
            
        Returns:
            bool: 同步模式下是否成功，异步模式下始终返回True
        """
        if not self.is_available:
            return False
            
        url = f"{self.base_url}/index/news"
        
        if async_mode:
            # 创建线程来执行请求
            thread = threading.Thread(
                target=self._make_request,
                args=(url, document)
            )
            thread.daemon = True
            thread.start()
            return True
        else:
            # 同步执行
            return self._make_request(url, document)
    
    def index_notice(self, document: Dict, async_mode: bool = True) -> bool:
        """
        索引公告到ES
        
        Args:
            document: 公告文档
            async_mode: 是否异步执行
            
        Returns:
            bool: 同步模式下是否成功，异步模式下始终返回True
        """
        if not self.is_available:
            return False
            
        url = f"{self.base_url}/index/notice"
        
        if async_mode:
            # 创建线程来执行请求
            thread = threading.Thread(
                target=self._make_request,
                args=(url, document)
            )
            thread.daemon = True
            thread.start()
            return True
        else:
            # 同步执行
            return self._make_request(url, document)
    
    def search_news(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        从ES搜索新闻
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            List[Dict]: 新闻列表
        """
        if not self.is_available:
            return []
            
        url = f"{self.base_url}/search/news"
        data = {"query": query, "n_results": n_results}
        
        try:
            response = self.session.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                app.logger.error(f"搜索新闻失败，状态码: {response.status_code}")
                return []
        except Exception as e:
            app.logger.error(f"搜索新闻时出错: {str(e)}")
            return []
    
    def search_notice(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        从ES搜索公告
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            List[Dict]: 公告列表
        """
        if not self.is_available:
            return []
            
        url = f"{self.base_url}/search/notice"
        data = {"query": query, "n_results": n_results}
        
        try:
            response = self.session.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                app.logger.error(f"搜索公告失败，状态码: {response.status_code}")
                return []
        except Exception as e:
            app.logger.error(f"搜索公告时出错: {str(e)}")
            return []
    
    def search_all(self, query: str, n_results: int = 5) -> Dict[str, List[Dict]]:
        """
        从ES同时搜索新闻和公告
        
        Args:
            query: 查询文本
            n_results: 每种类型返回的结果数量
            
        Returns:
            Dict[str, List[Dict]]: 包含新闻和公告的字典
        """
        if not self.is_available:
            return {"news": [], "announcements": []}
            
        url = f"{self.base_url}/search/all"
        data = {"query": query, "n_results": n_results}
        
        try:
            response = self.session.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                app.logger.error(f"搜索失败，状态码: {response.status_code}")
                return {"news": [], "announcements": []}
        except Exception as e:
            app.logger.error(f"搜索时出错: {str(e)}")
            return {"news": [], "announcements": []}
    
    def delete_news(self, doc_id: str) -> bool:
        """
        删除新闻
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 是否成功删除
        """
        if not self.is_available:
            return False
            
        url = f"{self.base_url}/delete/news/{doc_id}"
        
        try:
            response = self.session.delete(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            app.logger.error(f"删除新闻时出错: {str(e)}")
            return False
    
    def delete_notice(self, doc_id: str) -> bool:
        """
        删除公告
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 是否成功删除
        """
        if not self.is_available:
            return False
            
        url = f"{self.base_url}/delete/notice/{doc_id}"
        
        try:
            response = self.session.delete(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            app.logger.error(f"删除公告时出错: {str(e)}")
            return False
    
    def _make_request(self, url: str, data: Dict) -> bool:
        """
        执行请求
        
        Args:
            url: 请求URL
            data: 请求数据
            
        Returns:
            bool: 是否成功
        """
        try:
            response = self.session.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            app.logger.error(f"请求 {url} 时出错: {str(e)}")
            return False

class RankFusion:
    """结果融合算法工具类，用于融合向量和BM25搜索结果"""
    
    @staticmethod
    def contextual_fusion(query: str, dense_results: dict, lexical_results: dict, k: int = 60) -> dict:
        """
        上下文感知的融合算法，针对不同类型的查询动态调整权重
        
        Args:
            query: 用户查询
            dense_results: 向量检索结果 (格式: {"news": [...], "announcements": [...]})
            lexical_results: 文本检索结果 (格式: {"news": [...], "announcements": [...]})
            k: RRF常数
            
        Returns:
            融合后的结果字典 (格式: {"news": [...], "announcements": [...]})
        """
        # 提取查询特征
        query_terms = set(query.lower().split())
        is_status_query = any(term in query_terms for term in ['状态', '取消', '完成', '支付'])
        is_time_query = any(term in query_terms for term in ['时间', '日期', '年', '月', '日'])
        is_type_query = any(term in query_terms for term in ['类型', '种类', '分类'])
        
        # 动态调整权重
        if is_status_query or is_type_query:
            # 状态和类型查询，BM25可能更准确
            vector_weight = 0.4
            lexical_weight = 0.6
        elif is_time_query:
            # 时间查询，两者都重要
            vector_weight = 0.5
            lexical_weight = 0.5
        else:
            # 默认权重
            vector_weight = 0.7
            lexical_weight = 0.3
        
        # 用于保存融合结果
        fused_results = {"news": [], "announcements": []}
        
        # 处理新闻和公告
        for content_type in ["news", "announcements"]:
            # 获取各自的结果
            vector_content = dense_results.get(content_type, [])
            lexical_content = lexical_results.get(content_type, [])
            
            # 计算融合分数
            scores = {}
            
            # 处理向量结果
            for rank, item in enumerate(vector_content, start=1):
                item_id = item.get("id", "") 
                if not item_id:
                    continue
                    
                if item_id not in scores:
                    scores[item_id] = {"item": item, "score": 0, "matches": set()}
                
                scores[item_id]["score"] += vector_weight * (1.0 / (k + rank))
                scores[item_id]["matches"].add("vector")
            
            # 处理BM25结果
            for rank, item in enumerate(lexical_content, start=1):
                item_id = item.get("id", "")
                if not item_id:
                    continue
                    
                if item_id not in scores:
                    scores[item_id] = {"item": item, "score": 0, "matches": set()}
                
                scores[item_id]["score"] += lexical_weight * (1.0 / (k + rank))
                scores[item_id]["matches"].add("lexical")
                
                # 额外的上下文奖励
                content = item.get("content", "")
                title = item.get("title", "")
                
                # 检查结果是否包含查询词
                term_matches = sum(1 for term in query_terms if term in (content + title).lower())
                term_match_ratio = term_matches / len(query_terms) if query_terms else 0
                
                # 词匹配奖励
                scores[item_id]["score"] *= (1 + 0.2 * term_match_ratio)
            
            # 多检索源奖励
            for item_id, data in scores.items():
                if len(data["matches"]) > 1:  # 同时出现在两种检索中
                    data["score"] *= 1.25
            
            # 按分数排序
            sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
            fused_results[content_type] = [item_data["item"] for item_data in sorted_items]
        
        return fused_results

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
        self.status = {}   # 存储任务状态 (pending, running, completed, failed)
        self.callbacks = {} # 任务完成后的回调函数
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
            self.logger.info(f"启动工作线程 {i+1}")
    
    def submit_task(self, task_func: Callable, callback: Optional[Callable]=None, *args, **kwargs) -> str:
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
    
    def wait_for_task(self, task_id: str, timeout: Optional[float]=None) -> Any:
        """
        等待任务完成并返回结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            Any: 任务结果
            
        Raises:
            TimeoutError: 如果等待超时
            ValueError: 如果任务不存在
            RuntimeError: 如果任务执行失败
        """
        if task_id not in self.status:
            raise ValueError(f"任务 {task_id} 不存在")
            
        start_time = time.time()
        while self.status[task_id] in ["pending", "running"]:
            time.sleep(0.1)
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务 {task_id} 超时")
        
        if self.status[task_id] == "failed":
            raise RuntimeError(f"任务 {task_id} 执行失败: {self.results[task_id]}")
            
        return self.results[task_id]
    
    def clean_old_tasks(self, max_age: float=3600):
        """
        清理旧任务数据
        
        Args:
            max_age: 最大保留时间（秒），默认1小时
        """
        # 实现清理逻辑...
        pass

class DocumentProcessor:
    """Document processing utilities for different file types"""
    
    def __init__(self):
        """Initialize document processor with OCR model"""
        # Handle numpy compatibility issue
        if not hasattr(np, 'int'):
            np.int = np.int32
        
        # Initialize OCR model
        try:
            self.ocr = PaddleOCR(
                det_model_dir="/root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer",
                rec_model_dir="/root/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer",
                cls_model_dir="/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
                use_angle_cls=True,
                lang="ch"
            )
            app.logger.info("OCR model initialized successfully")
        except Exception as e:
            self.ocr = None
            app.logger.error(f"Failed to initialize OCR model: {str(e)}")
    
    def sanitize_html(self, html_content: str) -> str:
        """
        Sanitize HTML content to extract clean text
        
        Args:
            html_content: HTML string to sanitize
            
        Returns:
            str: Clean text extracted from HTML
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            app.logger.error(f"HTML sanitization error: {str(e)}")
            # Return the original content if sanitization fails
            return html_content

    def process_image(self, image_data: bytes) -> Tuple[str, float]:
        """
        Extract text from image using OCR
        
        Args:
            image_data: Binary image data
            
        Returns:
            Tuple[str, float]: Extracted text and average confidence score
        """
        if self.ocr is None:
            app.logger.error("OCR model not initialized")
            return "OCR model not available", 0.0
            
        try:
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(image_data))
            
            # Convert to numpy array
            img_np = np.array(img)
            
            # Run OCR
            result = self.ocr.ocr(img_np, cls=True)
            
            # Extract text and calculate average confidence
            text_parts = []
            confidence_sum = 0.0
            count = 0
            
            if result:
                for line in result:
                    if isinstance(line, list) and line and len(line) > 0:
                        if len(line[-1]) >= 2:
                            text = line[-1][0]  # Text content
                            confidence = float(line[-1][1])  # Confidence score
                            
                            if text and confidence > 0.5:  # Only include reasonably confident results
                                text_parts.append(text)
                                confidence_sum += confidence
                                count += 1
            
            # Calculate average confidence if there are valid detections
            avg_confidence = confidence_sum / count if count > 0 else 0.0
            full_text = "\n".join(text_parts)
            
            if not full_text.strip():
                return "No text detected in image", 0.0
                
            return full_text, avg_confidence
        
        except Exception as e:
            app.logger.error(f"Image processing error: {str(e)}")
            return f"Failed to process image: {str(e)}", 0.0

    def process_word_document(self, docx_data: bytes) -> str:
        """
        Extract text from Word document
        
        Args:
            docx_data: Binary Word document data
            
        Returns:
            str: Extracted text
        """
        try:
            result = mammoth.extract_raw_text(BytesIO(docx_data))
            return result.value
        except Exception as e:
            app.logger.error(f"Word document processing error: {str(e)}")
            return f"Failed to process Word document: {str(e)}"

    def process_pdf(self, pdf_data: bytes) -> str:
        """
        Extract text from PDF document
        
        Args:
            pdf_data: Binary PDF data
            
        Returns:
            str: Extracted text
        """
        try:
            with BytesIO(pdf_data) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text())
                
            return "\n\n".join(text)
        except Exception as e:
            app.logger.error(f"PDF processing error: {str(e)}")
            return f"Failed to process PDF: {str(e)}"

    def get_file_content(self, file_data: bytes, file_type: str) -> str:
        """
        Process file based on its type and return text content
        
        Args:
            file_data: Binary file data
            file_type: MIME type or file extension
            
        Returns:
            str: Extracted text content
        """
        file_type = file_type.lower()
        
        if file_type in ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/bmp']:
            text, confidence = self.process_image(file_data)
            return text
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword', '.docx', '.doc']:
            return self.process_word_document(file_data)
        elif file_type in ['application/pdf', '.pdf']:
            return self.process_pdf(file_data)
        elif file_type in ['text/html', '.html', '.htm']:
            return self.sanitize_html(file_data.decode('utf-8', errors='replace'))
        elif file_type in ['text/plain', '.txt']:
            return file_data.decode('utf-8', errors='replace')
        else:
            return f"Unsupported file type: {file_type}"

class ChineseRAGSystem:
    def __init__(
        self, 
        embedding_model_path: str = "/models/sentence-transformers_text2vec-large-chinese",
        llm_api_key: str = None,
        llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_model: str = "qwen-plus",
        chroma_host: str = None, 
        chroma_port: int = None,
        use_langchain: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_hybrid_search: bool = False,
        es_service_url: str = "http://localhost:8085"
    ):
        """
        初始化中文RAG系统
        
        Args:
            embedding_model_path: 嵌入模型的本地路径
            llm_api_key: 大模型API密钥
            llm_base_url: 大模型API基础URL
            llm_model: 使用的大模型名称
            chroma_host: ChromaDB服务器地址，如果为None则使用本地内存模式
            chroma_port: ChromaDB服务器端口
            use_langchain: 是否使用LangChain进行文档分块
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠部分大小
            use_hybrid_search: 是否开启混合检索(chromadb+bm25)
            es_service_url: es服务器
        """
        
        # ES服务配置
        self.use_hybrid_search = use_hybrid_search
        self.es_service_url = es_service_url
        
        # 初始化文档处理器
        self.doc_processor = DocumentProcessor()
        
        # 是否使用LangChain
        self.use_langchain = use_langchain
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 加载嵌入模型
        app.logger.info(f"正在加载嵌入模型: {embedding_model_path}...")
        self.embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_path,
            local_files_only=True
        )
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        app.logger.info(f"嵌入模型加载完成，向量维度：{self.vector_dim}")
        
        # 初始化LLM客户端
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.llm_client = OpenAI(
            api_key=self.llm_api_key,
            base_url=llm_base_url
        )
        self.llm_model = llm_model
        app.logger.info(f"LLM客户端初始化完成，使用模型: {llm_model}")
        
        # 初始化ES服务客户端（如果配置了ES服务URL）
        self.es_client = None
        if self.es_service_url and self.use_hybrid_search:
            self.es_client = ESServiceClient(base_url=self.es_service_url)
            app.logger.info(f"ES服务客户端初始化完成，服务可用: {self.es_client.is_available}")
        else:
            app.logger.info("未配置ES服务URL或未启用混合搜索，不初始化ES客户端")
        
        # 设置ChromaDB客户端
        try:
            if chroma_host and chroma_port:
                app.logger.info(f"尝试连接到远程ChromaDB: {chroma_host}:{chroma_port}")
                self.db_client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=Settings(anonymized_telemetry=False)
                )
                # 尝试一个简单操作来测试连接
                self.db_client.heartbeat()
                app.logger.info("成功连接到ChromaDB服务器")
            else:
                app.logger.info("使用内存模式ChromaDB")
                self.db_client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False)
                )
        except Exception as e:
            app.logger.error(f"连接ChromaDB时出错: {str(e)}")
            app.logger.info("回退到内存模式ChromaDB")
            self.db_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # 直接创建集合，不设置embedding_function，我们将手动计算嵌入向量
        try:
            self.news_collection = self.db_client.get_or_create_collection(
                name="cooper_news"
            )
            
            self.announcement_collection = self.db_client.get_or_create_collection(
                name="cooper_notice"
            )
            app.logger.info("成功创建/获取集合")
        except Exception as e:
            app.logger.error(f"创建集合时出错: {str(e)}")
            raise
            
        # 初始化LangChain文本分割器
        if self.use_langchain:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
    
    def compute_embeddings(self, texts):
        """计算文本的嵌入向量"""
        try:
            # 确保输入是列表
            if not isinstance(texts, list):
                texts = [texts]
                
            # 计算嵌入向量
            embeddings = self.embedding_model.encode(texts)
            
            # 转换为列表
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            elif isinstance(embeddings, list):
                # 检查是否是numpy数组的列表
                if embeddings and isinstance(embeddings[0], np.ndarray):
                    return [emb.tolist() for emb in embeddings]
                return embeddings
            else:
                app.logger.warning(f"Warning: Unexpected embedding type: {type(embeddings)}")
                return [[0.0] * self.vector_dim]  # 返回零向量作为后备
        except Exception as e:
            app.logger.error(f"计算嵌入向量时出错: {str(e)}")
            return [[0.0] * self.vector_dim]  # 返回零向量作为后备
    
    def split_text(self, text: str) -> List[str]:
        """
        将长文本分割成较小的块
        
        Args:
            text: 要分割的文本
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        if self.use_langchain:
            try:
                # 使用LangChain的文本分割器
                chunks = self.text_splitter.split_text(text)
                # 确保至少有一个块
                if not chunks:
                    chunks = [text]
                return chunks
            except Exception as e:
                app.logger.error(f"使用LangChain分割文本时出错: {str(e)}")
                # 回退到简单分割
                return [text]
        else:
            # 简单的基于段落的分割
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # 如果段落很少，直接返回
            if len(paragraphs) <= 1:
                return [text]
                
            # 合并短段落
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += (para + "\n\n")
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            # 添加最后一个块
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 确保至少有一个块
            if not chunks:
                chunks = [text]
                
            return chunks

    def search_news(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        搜索新闻
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 计算查询的嵌入向量
            query_embedding = self.compute_embeddings([query])[0]
            
            # 使用向量搜索
            results = self.news_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # 获取更多结果，因为可能有重复的base_id
                include=["metadatas", "documents", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                # 收集不同base_id的第一个结果
                seen_base_ids = set()
                
                for i in range(len(results["ids"][0])):
                    base_id = results["metadatas"][0][i].get("base_id", results["ids"][0][i])
                    
                    # 如果已经包含了这个base_id的文档，则跳过
                    if base_id in seen_base_ids:
                        continue
                    
                    seen_base_ids.add(base_id)
                    
                    formatted_results.append({
                        "id": base_id,
                        "title": results["metadatas"][0][i]["title"],
                        "source": results["metadatas"][0][i]["source"],
                        "publish_date": results["metadatas"][0][i]["publish_date"],
                        "content": results["documents"][0][i],
                        "relevance_score": 1 - float(results["distances"][0][i]) if results["distances"] else 0.0
                    })
                    
                    # 如果已经收集了足够的不同文档，就停止
                    if len(formatted_results) >= n_results:
                        break
            
            return formatted_results
        except Exception as e:
            app.logger.error(f"搜索新闻时出错: {str(e)}")
            return []
    
    def search_announcements(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        搜索公告
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 计算查询的嵌入向量
            query_embedding = self.compute_embeddings([query])[0]
            
            # 使用向量搜索
            results = self.announcement_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # 获取更多结果，因为可能有重复的base_id
                include=["metadatas", "documents", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                # 收集不同base_id的第一个结果
                seen_base_ids = set()
                
                for i in range(len(results["ids"][0])):
                    base_id = results["metadatas"][0][i].get("base_id", results["ids"][0][i])
                    
                    # 如果已经包含了这个base_id的文档，则跳过
                    if base_id in seen_base_ids:
                        continue
                    
                    seen_base_ids.add(base_id)
                    
                    formatted_results.append({
                        "id": base_id,
                        "title": results["metadatas"][0][i]["title"],
                        "department": results["metadatas"][0][i]["department"],
                        "publish_date": results["metadatas"][0][i]["publish_date"],
                        "importance": results["metadatas"][0][i]["importance"],
                        "content": results["documents"][0][i],
                        "relevance_score": 1 - float(results["distances"][0][i]) if results["distances"] else 0.0
                    })
                    
                    # 如果已经收集了足够的不同文档，就停止
                    if len(formatted_results) >= n_results:
                        break
            
            return formatted_results
        except Exception as e:
            app.logger.error(f"搜索公告时出错: {str(e)}")
            return []
    
    def search_all(self, query: str, n_results: int = 5) -> Dict[str, List[Dict]]:
        """
        同时搜索新闻和公告
        
        Args:
            query: 查询文本
            n_results: 每种类型返回的结果数量
            
        Returns:
            Dict[str, List[Dict]]: 搜索结果字典，包含新闻和公告
        """
        news_results = self.search_news(query, n_results)
        announcement_results = self.search_announcements(query, n_results)
        
        return {
            "news": news_results,
            "announcements": announcement_results
        }
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        根据查询和上下文生成回答
        
        Args:
            query: 用户查询
            context: 检索的上下文内容
            temperature: 温度参数，控制回答的创造性，值越高越创造性
            max_tokens: 最大生成token数，限制回答长度
            
        Returns:
            str: 生成的回答
        """
        try:
            # 设置系统提示，指导大模型的行为
            system_prompt = """
            你是一个专业的中文新闻与公告智能助手。请严格基于提供的上下文信息回答问题，不要添加任何未在上下文中明确提到的信息。
            回答要求：
            1. 简洁明了：保持回答简洁、结构清晰，重点突出
            2. 信息归因：引用信息时指明来源（例如"根据XX新闻报道/XX公告通知..."）
            3. 处理不确定性：如果上下文信息不足或存在矛盾，明确指出并说明限制
            4. 时效性标注：提及日期和时间信息时，注明信息的时间背景
            5. 区分处理：新闻内容以客观陈述为主，公告内容需强调其官方性和指导意义

            当无法从上下文中找到相关信息时，请直接回答："根据现有信息，我无法回答这个问题。请问您是否想了解我们系统中的其他新闻或公告？"

            对于复杂询问，先分析问题的核心需求，再从上下文提取相关信息，确保回答全面且准确。
            """
            
            # 设置用户提示，包含查询和上下文
            user_prompt = f"""用户问题: {query}
                        
                ----上下文信息----
                {context}
                ----上下文信息结束----

                基于上述上下文信息，请回答用户的问题。如果上下文信息不足以回答用户问题，请明确指出。"""

            # 调用大语言模型API生成回答
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,  # 使用配置的模型，如"qwen-plus"
                messages=[
                    {"role": "system", "content": system_prompt},  # 系统角色提示
                    {"role": "user", "content": user_prompt}       # 用户角色提示
                ],
                temperature=temperature,  # 控制多样性
                max_tokens=max_tokens     # 控制回答长度
            )
            
            # 提取并返回生成的回答内容
            return response.choices[0].message.content
        
        except Exception as e:
            # 异常处理：记录错误并返回错误信息
            app.logger.error(f"生成回答时出错: {str(e)}")
            return f"生成回答时发生错误: {str(e)}"
           
    def format_context(self, search_results: Dict[str, List[Dict]]) -> str:
        """
        将搜索结果格式化为上下文信息，用于LLM输入
        
        Args:
            search_results: 搜索结果
            
        Returns:
            str: 格式化后的上下文
        """
        context = []
        
        # 添加新闻
        if search_results["news"]:
            context.append("## 相关新闻")
            for i, news in enumerate(search_results["news"]):
                context.append(f"{i+1}. 标题: {news['title']}")
                context.append(f"   来源: {news['source']} ({news['publish_date']})")
                context.append(f"   内容: {news['content']}")
                context.append("")
        
        # 添加公告
        if search_results["announcements"]:
            context.append("## 相关公告")
            for i, announcement in enumerate(search_results["announcements"]):
                importance_marker = "🔴" if announcement['importance'] == "high" else "🟢"
                context.append(f"{i+1}. {importance_marker} {announcement['title']}")
                context.append(f"   发布: {announcement['department']} ({announcement['publish_date']})")
                context.append(f"   内容: {announcement['content']}")
                context.append("")
        
        if not context:
            return "未找到相关信息。"
            
        return "\n".join(context)
            
    def query(
        self, 
        query: str, 
        n_results: int = 3,
        temperature: float = 0.7, 
        max_tokens: int = 1000,
        use_hybrid_search: bool = None  # 可选参数，默认使用配置值
    ) -> Dict:
        """
        端到端RAG查询流程
        
        Args:
            query: 用户查询
            n_results: 每类检索的结果数量
            temperature: LLM温度参数
            max_tokens: 最大生成token数
            use_hybrid_search: 是否使用混合搜索（向量+BM25），None表示使用配置值
            
        Returns:
            Dict: 包含检索结果和生成的回答
        """
        # 1. 检索相关文档
        try:
            # 确定是否使用混合搜索
            if use_hybrid_search is None:
                use_hybrid_search = self.use_hybrid_search and self.es_client and self.es_client.is_available
            
            # 执行检索
            if use_hybrid_search and self.es_client and self.es_client.is_available:
                search_results = self.hybrid_search_all(query, n_results)
                app.logger.info(f"使用混合搜索（向量+BM25）检索结果")
            else:
                search_results = self.search_all(query, n_results)
                app.logger.info(f"使用纯向量检索结果")
            
            # 2. 格式化上下文
            context = self.format_context(search_results)
            
            # 3. 生成回答
            answer = self.generate_response(
                query=query,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 4. 返回结果
            return {
                "query": query,
                "search_results": search_results,
                "context": context,
                "answer": answer,
                "search_type": "hybrid" if (use_hybrid_search and self.es_client and self.es_client.is_available) else "vector"
            }
        except Exception as e:
            app.logger.error(f"查询过程中发生错误: {str(e)}")
            # 返回基本响应
            return {
                "query": query,
                "search_results": {"news": [], "announcements": []},
                "context": "查询处理过程中发生错误",
                "answer": f"很抱歉，在处理您的查询时发生了错误: {str(e)}。请稍后再试或联系管理员。",
                "search_type": "error"
            }

    def add_news_async(self, 
                    title: str, 
                    content: str, 
                    source: str = None, 
                    publish_date: str = None, 
                    tags: List[str] = None, 
                    id: str = None) -> str:
        """
        异步添加新闻文章，处理HTML内容
        
        Args:
            title: 新闻标题
            content: 新闻正文（可能包含HTML）
            source: 新闻来源
            publish_date: 发布日期（格式：YYYY-MM-DD）
            tags: 标签列表
            id: 唯一ID，如果未提供则自动生成
            
        Returns:
            str: 任务ID
        """
        # 如果没有提供ID，生成一个
        base_id = id or str(uuid.uuid4())
        
        # 准备要在线程中执行的任务函数
        def process_and_add_task():
            try:
                original_content = content
                processed_content = original_content
                
                # 检查content是否包含HTML内容
                if '<' in original_content and '>' in original_content:
                    try:
                        # 使用BeautifulSoup解析HTML内容
                        soup = BeautifulSoup(original_content, 'html.parser')
                        
                        # 获取纯文本内容
                        text_content = self.doc_processor.sanitize_html(original_content)
                        
                        # 处理嵌入的图片
                        embedded_contents = []
                        for img in soup.find_all('img'):
                            src = img.get('src', '')
                            if src and (src.startswith('http://') or src.startswith('https://')):
                                try:
                                    # 下载图片
                                    img_response = requests.get(src, timeout=10)
                                    if img_response.status_code == 200:
                                        # 处理图片中的文本
                                        img_text, confidence = self.doc_processor.process_image(img_response.content)
                                        if img_text and img_text != "No text detected in image":
                                            embedded_contents.append(f"【图片内容】: {img_text}")
                                except Exception as e:
                                    app.logger.warning(f"处理嵌入图片时出错: {str(e)}")
                        
                        # 处理嵌入的文档链接
                        for a in soup.find_all('a'):
                            href = a.get('href', '')
                            if href and (href.endswith('.pdf') or href.endswith('.docx') or href.endswith('.doc')):
                                try:
                                    # 下载文档
                                    doc_response = requests.get(href, timeout=20)
                                    if doc_response.status_code == 200:
                                        # 根据文件类型处理文档
                                        if href.endswith('.pdf'):
                                            doc_text = self.doc_processor.process_pdf(doc_response.content)
                                            embedded_contents.append(f"【PDF文档内容】: {doc_text}")
                                        elif href.endswith('.docx') or href.endswith('.doc'):
                                            doc_text = self.doc_processor.process_word_document(doc_response.content)
                                            embedded_contents.append(f"【Word文档内容】: {doc_text}")
                                except Exception as e:
                                    app.logger.warning(f"处理嵌入文档时出错: {str(e)}")
                        
                        # 组合所有内容
                        if embedded_contents:
                            extracted_content = "\n\n".join(embedded_contents)
                            processed_content = f"{text_content}\n\n{extracted_content}"
                        else:
                            processed_content = text_content
                            
                    except Exception as e:
                        app.logger.error(f"处理HTML内容时出错: {str(e)}")
                        # 失败时至少清理HTML标签
                        processed_content = self.doc_processor.sanitize_html(original_content)
                
                # 准备元数据
                metadata = {
                    "title": title,
                    "source": source or "未知来源",
                    "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                    "type": "news",
                }
                
                if tags:
                    if isinstance(tags, str):
                        metadata["tags"] = tags
                    else:
                        metadata["tags"] = ",".join(tags)
                
                # 分割长文本
                document_chunks = self.split_text(processed_content)
                app.logger.info(f"将新闻文章 '{title}' 分割为 {len(document_chunks)} 个块")
                
                all_doc_ids = []
                
                for i, chunk in enumerate(document_chunks):
                    # 为每个块生成唯一ID
                    doc_id = f"{base_id}_{i}" if len(document_chunks) > 1 else base_id
                    
                    # 准备文档内容 (为块添加标题以保持上下文)
                    document = f"{title}\n{chunk}"
                    
                    # 为这个块更新元数据
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(document_chunks)
                    chunk_metadata["base_id"] = base_id
                    
                    # 计算嵌入向量
                    embeddings = self.compute_embeddings([document])
                    
                    # 添加到集合
                    try:
                        self.news_collection.add(
                            documents=[document],
                            metadatas=[chunk_metadata],
                            ids=[doc_id],
                            embeddings=embeddings
                        )
                        all_doc_ids.append(doc_id)
                    except Exception as e:
                        app.logger.error(f"添加新闻块 {i+1}/{len(document_chunks)} 时出错: {str(e)}")
                        # 继续添加其他块
                        continue
                
                if all_doc_ids:
                    app.logger.info(f"成功添加新闻: {title}，分为 {len(all_doc_ids)}/{len(document_chunks)} 个块")
                    if self.es_client and self.es_client.is_available:
                        for i, chunk in enumerate(document_chunks):
                            doc_id = f"{base_id}_{i}" if len(document_chunks) > 1 else base_id
                            
                            # 准备ES文档
                            es_document = {
                                "id": doc_id,
                                "title": title,
                                "content": chunk,
                                "source": source or "未知来源",
                                "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                                "base_id": base_id,
                                "chunk_index": i,
                                "total_chunks": len(document_chunks)
                            }
                            
                            if tags:
                                if isinstance(tags, list):
                                    es_document["tags"] = ",".join(tags)
                                else:
                                    es_document["tags"] = tags
                            
                            # 异步索引到ES (不要在线程中创建新线程，直接调用)
                            self.es_client.index_news(es_document, async_mode=True)
                    

                    app.logger.info(f"异步成功添加新闻: {title}，已同时索引到ES服务")
                    
                    return base_id
                else:
                    raise Exception("添加新闻时所有块都失败")
            
            except Exception as e:
                app.logger.error(f"异步添加新闻时出错: {str(e)}")
                raise
        
        # 提交到任务管理器
        task_id = task_manager.submit_task(process_and_add_task)
        app.logger.info(f"提交添加新闻任务: {task_id}, 文档基础ID: {base_id}")
        
        return task_id

    def add_announcement_async(self, 
                            title: str, 
                            content: str, 
                            department: str = None, 
                            publish_date: str = None,
                            importance: str = "normal",
                            id: str = None) -> str:
        """
        异步添加公告，处理HTML内容
        
        Args:
            title: 公告标题
            content: 公告内容（可能包含HTML）
            department: 发布部门
            publish_date: 发布日期（格式：YYYY-MM-DD）
            importance: 重要性（high, normal, low）
            id: 唯一ID，如果未提供则自动生成
            
        Returns:
            str: 任务ID
        """
        # 如果没有提供ID，生成一个
        base_id = id or str(uuid.uuid4())
        
        # 准备要在线程中执行的任务函数
        def process_and_add_task():
            try:
                original_content = content
                processed_content = original_content
                
                # 检查content是否包含HTML内容
                if '<' in original_content and '>' in original_content:
                    try:
                        # 使用BeautifulSoup解析HTML内容
                        soup = BeautifulSoup(original_content, 'html.parser')
                        
                        # 获取纯文本内容
                        text_content = self.doc_processor.sanitize_html(original_content)
                        
                        # 处理嵌入的图片
                        embedded_contents = []
                        for img in soup.find_all('img'):
                            src = img.get('src', '')
                            if src and (src.startswith('http://') or src.startswith('https://')):
                                try:
                                    # 下载图片
                                    img_response = requests.get(src, timeout=10)
                                    if img_response.status_code == 200:
                                        # 处理图片中的文本
                                        img_text, confidence = self.doc_processor.process_image(img_response.content)
                                        if img_text and img_text != "No text detected in image":
                                            embedded_contents.append(f"【图片内容】: {img_text}")
                                except Exception as e:
                                    app.logger.warning(f"处理嵌入图片时出错: {str(e)}")
                        
                        # 处理嵌入的文档链接
                        for a in soup.find_all('a'):
                            href = a.get('href', '')
                            if href and (href.endswith('.pdf') or href.endswith('.docx') or href.endswith('.doc')):
                                try:
                                    # 下载文档
                                    doc_response = requests.get(href, timeout=20)
                                    if doc_response.status_code == 200:
                                        # 根据文件类型处理文档
                                        if href.endswith('.pdf'):
                                            doc_text = self.doc_processor.process_pdf(doc_response.content)
                                            embedded_contents.append(f"【PDF文档内容】: {doc_text}")
                                        elif href.endswith('.docx') or href.endswith('.doc'):
                                            doc_text = self.doc_processor.process_word_document(doc_response.content)
                                            embedded_contents.append(f"【Word文档内容】: {doc_text}")
                                except Exception as e:
                                    app.logger.warning(f"处理嵌入文档时出错: {str(e)}")
                        
                        # 组合所有内容
                        if embedded_contents:
                            extracted_content = "\n\n".join(embedded_contents)
                            processed_content = f"{text_content}\n\n{extracted_content}"
                        else:
                            processed_content = text_content
                            
                    except Exception as e:
                        app.logger.error(f"处理HTML内容时出错: {str(e)}")
                        # 失败时至少清理HTML标签
                        processed_content = self.doc_processor.sanitize_html(original_content)
                
                # 准备元数据
                metadata = {
                    "title": title,
                    "department": department or "未知部门",
                    "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                    "importance": importance,
                    "type": "announcement",
                }
                
                # 分割长文本
                document_chunks = self.split_text(processed_content)
                app.logger.info(f"将公告 '{title}' 分割为 {len(document_chunks)} 个块")
                
                all_doc_ids = []
                
                for i, chunk in enumerate(document_chunks):
                    # 为每个块生成唯一ID
                    doc_id = f"{base_id}_{i}" if len(document_chunks) > 1 else base_id
                    
                    # 准备文档内容 (为块添加标题以保持上下文)
                    document = f"{title}\n{chunk}"
                    
                    # 为这个块更新元数据
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(document_chunks)
                    chunk_metadata["base_id"] = base_id
                    
                    # 计算嵌入向量
                    embeddings = self.compute_embeddings([document])
                    
                    # 添加到集合
                    try:
                        self.announcement_collection.add(
                            documents=[document],
                            metadatas=[chunk_metadata],
                            ids=[doc_id],
                            embeddings=embeddings
                        )
                        all_doc_ids.append(doc_id)
                    except Exception as e:
                        app.logger.error(f"添加公告块 {i+1}/{len(document_chunks)} 时出错: {str(e)}")
                        # 继续添加其他块
                        continue
                
                if all_doc_ids:
                    app.logger.info(f"成功添加公告: {title}，分为 {len(all_doc_ids)}/{len(document_chunks)} 个块")
                    if self.es_client and self.es_client.is_available:
                        for i, chunk in enumerate(document_chunks):
                            doc_id = f"{base_id}_{i}" if len(document_chunks) > 1 else base_id
                            
                            # 准备ES文档
                            es_document = {
                                "id": doc_id,
                                "title": title,
                                "content": chunk,
                                "department": department or "未知部门",
                                "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
                                "importance": importance,
                                "base_id": base_id,
                                "chunk_index": i,
                                "total_chunks": len(document_chunks)
                            }
                            
                            # 异步索引到ES (不要在线程中创建新线程，直接调用)
                            self.es_client.index_notice(es_document, async_mode=True)

                    app.logger.info(f"异步成功添加公告: {title}，已同时索引到ES服务")
                    return base_id
                else:
                    raise Exception("添加公告时所有块都失败")
            
            except Exception as e:
                app.logger.error(f"异步添加公告时出错: {str(e)}")
                raise
        
        # 提交到任务管理器
        task_id = task_manager.submit_task(process_and_add_task)
        app.logger.info(f"提交添加公告任务: {task_id}, 文档基础ID: {base_id}")
        
        return task_id

    def hybrid_search_all(self, query: str, n_results: int = 5) -> Dict[str, List[Dict]]:
        """
        混合搜索：使用向量检索和BM25融合搜索结果
        
        Args:
            query: 查询文本
            n_results: 每种类型返回的结果数量
            
        Returns:
            Dict[str, List[Dict]]: 融合后的搜索结果
        """
        # 1. 向量检索
        vector_results = self.search_all(query, n_results)
        
        # 2. 如果ES服务可用，执行BM25检索
        if self.es_client and self.es_client.is_available:
            bm25_results = self.es_client.search_all(query, n_results)
            
            # 3. 融合结果
            try:
                fused_results = RankFusion.contextual_fusion(
                    query=query,
                    dense_results=vector_results, 
                    lexical_results=bm25_results
                )
                return fused_results
            except Exception as e:
                app.logger.error(f"融合搜索结果时出错: {str(e)}")
                # 出错时回退到向量检索结果
                return vector_results
        
        # 如果ES服务不可用，直接返回向量检索结果
        return vector_results

# 初始化RAG系统
# 从环境变量获取配置
EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', '/models/sentence-transformers_text2vec-large-chinese')
LLM_API_KEY = os.environ.get('LLM_API_KEY', 'your_openai_key_here')
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen-plus')
CHROMA_HOST = os.environ.get('CHROMA_HOST','192.168.222.128')
CHROMA_PORT = os.environ.get('CHROMA_PORT','8000')
USE_LANGCHAIN = os.environ.get('USE_LANGCHAIN', 'true').lower() == 'true'
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))
USE_HYBRID_SEARCH = os.environ.get('use_hybrid_search', 'true').lower() == 'true'
ES_SERVICE_URL = os.environ.get('es_service_url', 'http://192.168.222.128:8085')

if CHROMA_PORT and CHROMA_PORT.isdigit():
    CHROMA_PORT = int(CHROMA_PORT)
else:
    CHROMA_PORT = None

# 应用启动时初始化RAG系统
rag_system = None

# 替换之前的 @app.before_first_request
with app.app_context():
    try:
        app.logger.info("初始化RAG系统...")
        rag_system = ChineseRAGSystem(
            embedding_model_path=EMBEDDING_MODEL_PATH,
            llm_api_key=LLM_API_KEY,
            llm_base_url=LLM_BASE_URL,
            llm_model=LLM_MODEL,
            chroma_host=CHROMA_HOST,
            chroma_port=CHROMA_PORT,
            use_langchain=USE_LANGCHAIN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_hybrid_search = USE_HYBRID_SEARCH,
            es_service_url = ES_SERVICE_URL if USE_HYBRID_SEARCH else None
        )

        # 检查ES服务状态
        if USE_HYBRID_SEARCH:
            if rag_system.es_client and rag_system.es_client.is_available:
                app.logger.info(f"ES服务可用，混合搜索已启用")
            else:
                app.logger.warning(f"ES服务不可用，将使用纯向量搜索")
        
        app.logger.info("RAG系统初始化完成")
    except Exception as e:
        app.logger.error(f"初始化RAG系统时出错: {str(e)}")


# 创建任务管理器
task_manager = AsyncTaskManager(max_workers=3)  # 设置3个工作线程

# 注册路由
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "use_langchain": USE_LANGCHAIN,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "numpy_version": np.__version__
        }
    })

@app.route('/query', methods=['POST'])
def query_endpoint():
    """查询接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "请提供查询内容"}), 400
    
    query = data['query']
    n_results = data.get('n_results', 3)
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 1000)
    
    try:
        result = rag_system.query(
            query=query,
            n_results=n_results,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # 使用ensure_ascii=False确保中文字符不会被编码成Unicode转义序列
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        app.logger.error(f"处理查询请求时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/add/news', methods=['POST'])
def add_news_endpoint():
    """添加新闻接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    # 检查表单或JSON数据
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()
    
    # 验证必要字段
    if not data or 'title' not in data:
        return jsonify({"error": "请提供标题"}), 400
    
    if 'content' not in data:
        return jsonify({"error": "请提供内容"}), 400
    
    try:
        # 准备标签
        tags = None
        if 'tags' in data and data['tags']:
            if isinstance(data['tags'], str):
                tags = [tag.strip() for tag in data['tags'].split(',') if tag.strip()]
            elif isinstance(data['tags'], list):
                tags = data['tags']
        
        # 异步添加新闻
        task_id = rag_system.add_news_async(
            title=data['title'],
            content=data['content'],
            source=data.get('source'),
            publish_date=data.get('publish_date'),
            tags=tags
        )
        
        # 返回任务ID和成功消息
        return jsonify({
            "success": True, 
            "message": "新闻正在异步处理中，包括HTML内容提取和嵌入图片/文档的处理",
            "task_id": task_id
        })
    except Exception as e:
        app.logger.error(f"添加新闻时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/add/announcement', methods=['POST'])
def add_announcement_endpoint():
    """添加公告接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    # 检查表单或JSON数据
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()
    
    # 验证必要字段
    if not data or 'title' not in data:
        return jsonify({"error": "请提供标题"}), 400
    
    if 'content' not in data:
        return jsonify({"error": "请提供内容"}), 400
    
    try:
        # 异步添加公告
        task_id = rag_system.add_announcement_async(
            title=data['title'],
            content=data['content'],
            department=data.get('department'),
            publish_date=data.get('publish_date'),
            importance=data.get('importance', 'normal')
        )
        
        # 返回任务ID和成功消息
        return jsonify({
            "success": True, 
            "message": "公告正在异步处理中，包括HTML内容提取和嵌入图片/文档的处理",
            "task_id": task_id
        })
    except Exception as e:
        app.logger.error(f"添加公告时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/task/status/<task_id>', methods=['GET'])
def task_status_endpoint(task_id):
    """查询任务状态接口"""
    status = task_manager.get_task_status(task_id)
    return jsonify(status)

@app.route('/delete/news/<doc_id>', methods=['DELETE'])
def delete_news_endpoint(doc_id):
    """删除指定ID的新闻"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    try:
        # 检查doc_id是否包含块索引
        if '_' in doc_id:
            base_id = doc_id.split('_')[0]
            # 查找所有相关块
            query = f"base_id:'{base_id}'"
            results = rag_system.news_collection.get(
                where={"base_id": base_id}
            )
            
            if not results["ids"]:
                return jsonify({"error": f"找不到ID为{base_id}的新闻"}), 404
            
            # 删除所有相关块
            rag_system.news_collection.delete(ids=results["ids"])
            app.logger.info(f"成功删除新闻ID: {base_id} 的所有 {len(results['ids'])} 个块")
            
            return jsonify({"success": True, "message": f"成功删除新闻ID: {base_id} 的所有 {len(results['ids'])} 个块"})
        else:
            # 对于单块文档或尝试删除所有相关块
            # 首先检查是否是父ID
            results = rag_system.news_collection.get(
                where={"base_id": doc_id}
            )
            
            if results["ids"]:
                # 这是一个父ID，删除所有相关块
                rag_system.news_collection.delete(ids=results["ids"])
                app.logger.info(f"成功删除新闻ID: {doc_id} 的所有 {len(results['ids'])} 个块")
                return jsonify({"success": True, "message": f"成功删除新闻ID: {doc_id} 的所有 {len(results['ids'])} 个块"})
            else:
                # 尝试直接删除该ID
                direct_results = rag_system.news_collection.get(ids=[doc_id], include=[])
                if not direct_results["ids"]:
                    return jsonify({"error": f"找不到ID为{doc_id}的新闻"}), 404
                
                # 删除文档
                rag_system.news_collection.delete(ids=[doc_id])
                app.logger.info(f"成功删除新闻ID: {doc_id}")
                
                return jsonify({"success": True, "message": f"成功删除新闻ID: {doc_id}"})
    except Exception as e:
        app.logger.error(f"删除新闻时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete/announcement/<doc_id>', methods=['DELETE'])
def delete_announcement_endpoint(doc_id):
    """删除指定ID的公告"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    try:
        # 检查doc_id是否包含块索引
        if '_' in doc_id:
            base_id = doc_id.split('_')[0]
            # 查找所有相关块
            results = rag_system.announcement_collection.get(
                where={"base_id": base_id}
            )
            
            if not results["ids"]:
                return jsonify({"error": f"找不到ID为{base_id}的公告"}), 404
            
            # 删除所有相关块
            rag_system.announcement_collection.delete(ids=results["ids"])
            app.logger.info(f"成功删除公告ID: {base_id} 的所有 {len(results['ids'])} 个块")
            
            return jsonify({"success": True, "message": f"成功删除公告ID: {base_id} 的所有 {len(results['ids'])} 个块"})
        else:
            # 对于单块文档或尝试删除所有相关块
            # 首先检查是否是父ID
            results = rag_system.announcement_collection.get(
                where={"base_id": doc_id}
            )
            
            if results["ids"]:
                # 这是一个父ID，删除所有相关块
                rag_system.announcement_collection.delete(ids=results["ids"])
                app.logger.info(f"成功删除公告ID: {doc_id} 的所有 {len(results['ids'])} 个块")
                return jsonify({"success": True, "message": f"成功删除公告ID: {doc_id} 的所有 {len(results['ids'])} 个块"})
            else:
                # 尝试直接删除该ID
                direct_results = rag_system.announcement_collection.get(ids=[doc_id], include=[])
                if not direct_results["ids"]:
                    return jsonify({"error": f"找不到ID为{doc_id}的公告"}), 404
                
                # 删除文档
                rag_system.announcement_collection.delete(ids=[doc_id])
                app.logger.info(f"成功删除公告ID: {doc_id}")
                
                return jsonify({"success": True, "message": f"成功删除公告ID: {doc_id}"})
    except Exception as e:
        app.logger.error(f"删除公告时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process/file', methods=['POST'])
def process_file_endpoint():
    """处理文件提取内容接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "请提供文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    try:
        file_data = file.read()
        file_type = file.content_type or os.path.splitext(file.filename)[1]
        
        extracted_content = rag_system.doc_processor.get_file_content(file_data, file_type)
        
        return jsonify({
            "success": True, 
            "filename": file.filename,
            "file_type": file_type,
            "content": extracted_content,
            "content_length": len(extracted_content)
        })
    except Exception as e:
        app.logger.error(f"处理文件时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process/html', methods=['POST'])
def process_html_endpoint():
    """处理HTML内容接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    data = request.json
    if not data or 'html' not in data:
        return jsonify({"error": "请提供HTML内容"}), 400
    
    try:
        html_content = data['html']
        sanitized_content = rag_system.doc_processor.sanitize_html(html_content)
        
        return jsonify({
            "success": True,
            "content": sanitized_content,
            "content_length": len(sanitized_content)
        })
    except Exception as e:
        app.logger.error(f"处理HTML内容时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/hybrid_query', methods=['POST'])
def hybrid_query_endpoint():
    """混合查询接口（向量+BM25）"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "请提供查询内容"}), 400
    
    query = data['query']
    n_results = data.get('n_results', 3)
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 1000)
    
    try:
        result = rag_system.query(
            query=query,
            n_results=n_results,
            temperature=temperature,
            max_tokens=max_tokens,
            use_hybrid_search=True  # 强制使用混合搜索
        )
        # 使用ensure_ascii=False确保中文字符不会被编码成Unicode转义序列
        return app.response_class(
            response=json.dumps(result, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        app.logger.error(f"处理混合查询请求时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/es_status', methods=['GET'])
def es_status_endpoint():
    """ES服务状态接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    if rag_system.es_client:
        is_available = rag_system.es_client.is_available
        return jsonify({
            "es_service_enabled": True,
            "es_service_available": is_available,
            "es_service_url": rag_system.es_service_url,
            "hybrid_search_enabled": rag_system.use_hybrid_search
        })
    else:
        return jsonify({
            "es_service_enabled": False,
            "es_service_available": False,
            "hybrid_search_enabled": rag_system.use_hybrid_search
        })
        
# 启动函数
if __name__ == "__main__":
    # 如果存在PORT环境变量，使用它，否则使用默认的5000
    port = int(os.environ.get("PORT", 5000))
    # 在开发模式下启用调试，生产环境应使用 gunicorn
    debug = os.environ.get("FLASK_ENV") == "development"
    
    app.run(host="0.0.0.0", port=port, debug=debug)
        