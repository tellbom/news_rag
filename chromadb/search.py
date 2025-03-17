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

from flask import Flask
app = Flask(__name__)

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
        chunk_overlap: int = 200
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
        """
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
    
    def add_news(self, 
                title: str, 
                content: str, 
                source: str = None, 
                publish_date: str = None, 
                tags: List[str] = None, 
                id: str = None,
                file_data: bytes = None,
                file_type: str = None) -> str:
        """
        添加新闻文章
        
        Args:
            title: 新闻标题
            content: 新闻正文
            source: 新闻来源
            publish_date: 发布日期（格式：YYYY-MM-DD）
            tags: 标签列表
            id: 唯一ID，如果未提供则自动生成
            file_data: 文件数据
            file_type: 文件类型
            
        Returns:
            str: 文档ID
        """
        # 如果提供了文件，处理文件内容
        if file_data and file_type:
            file_content = self.doc_processor.get_file_content(file_data, file_type)
            # 如果content为空，直接使用文件内容
            if not content or content.strip() == "":
                content = file_content
            # 否则将文件内容添加到现有内容
            else:
                content = f"{content}\n\n{file_content}"
        
        # 准备元数据
        metadata = {
            "title": title,
            "source": source or "未知来源",
            "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
            "type": "news",
        }
        
        if tags:
            metadata["tags"] = ",".join(tags)
        
        # 如果没有提供ID，生成一个
        base_id = id or str(uuid.uuid4())
        
        # 分割长文本
        document_chunks = self.split_text(content)
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
            return base_id
        else:
            raise Exception("添加新闻时所有块都失败")
    
    def add_announcement(self, 
                       title: str, 
                       content: str, 
                       department: str = None, 
                       publish_date: str = None,
                       importance: str = "normal",
                       id: str = None,
                       file_data: bytes = None,
                       file_type: str = None) -> str:
        """
        添加公告
        
        Args:
            title: 公告标题
            content: 公告内容
            department: 发布部门
            publish_date: 发布日期（格式：YYYY-MM-DD）
            importance: 重要性（high, normal, low）
            id: 唯一ID，如果未提供则自动生成
            file_data: 文件数据
            file_type: 文件类型
            
        Returns:
            str: 文档ID
        """
        # 如果提供了文件，处理文件内容
        if file_data and file_type:
            file_content = self.doc_processor.get_file_content(file_data, file_type)
            # 如果content为空，直接使用文件内容
            if not content or content.strip() == "":
                content = file_content
            # 否则将文件内容添加到现有内容
            else:
                content = f"{content}\n\n{file_content}"
        
        # 准备元数据
        metadata = {
            "title": title,
            "department": department or "未知部门",
            "publish_date": publish_date or datetime.now().strftime("%Y-%m-%d"),
            "importance": importance,
            "type": "announcement",
        }
        
        # 如果没有提供ID，生成一个
        base_id = id or str(uuid.uuid4())
        
        # 分割长文本
        document_chunks = self.split_text(content)
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
            return base_id
        else:
            raise Exception("添加公告时所有块都失败")
    
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
            system_prompt = """你是一个基于检索增强生成(RAG)的智能助手。你的回答将基于提供的上下文信息。
            如果上下文中没有足够的信息，请坦诚地表示你不知道，不要编造信息。
            如果用户询问的内容与上下文无关，请礼貌地引导用户提问与新闻或公告相关的问题。"""
            
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
        max_tokens: int = 1000
    ) -> Dict:
        """
        端到端RAG查询流程
        
        Args:
            query: 用户查询
            n_results: 每类检索的结果数量
            temperature: LLM温度参数
            max_tokens: 最大生成token数
            
        Returns:
            Dict: 包含检索结果和生成的回答
        """
        # 1. 检索相关文档
        try:
            search_results = self.search_all(query, n_results)
            
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
                "answer": answer
            }
        except Exception as e:
            app.logger.error(f"查询过程中发生错误: {str(e)}")
            # 返回基本响应
            return {
                "query": query,
                "search_results": {"news": [], "announcements": []},
                "context": "查询处理过程中发生错误",
                "answer": f"很抱歉，在处理您的查询时发生了错误: {str(e)}。请稍后再试或联系管理员。"
            }

   

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
            chunk_overlap=CHUNK_OVERLAP
        )
        app.logger.info("RAG系统初始化完成")
    except Exception as e:
        app.logger.error(f"初始化RAG系统时出错: {str(e)}")


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
    
    original_content = data['content']
    processed_content = original_content
    
    # 检查content是否包含HTML内容
    if '<' in original_content and '>' in original_content:
        try:
            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(original_content, 'html.parser')
            
            # 获取纯文本内容
            text_content = rag_system.doc_processor.sanitize_html(original_content)
            
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
                            img_text, confidence = rag_system.doc_processor.process_image(img_response.content)
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
                                doc_text = rag_system.doc_processor.process_pdf(doc_response.content)
                                embedded_contents.append(f"【PDF文档内容】: {doc_text}")
                            elif href.endswith('.docx') or href.endswith('.doc'):
                                doc_text = rag_system.doc_processor.process_word_document(doc_response.content)
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
            processed_content = rag_system.doc_processor.sanitize_html(original_content)
    
    try:
        # 准备标签
        tags = None
        if 'tags' in data and data['tags']:
            if isinstance(data['tags'], str):
                tags = [tag.strip() for tag in data['tags'].split(',') if tag.strip()]
            elif isinstance(data['tags'], list):
                tags = data['tags']
        
        doc_id = rag_system.add_news(
            title=data['title'],
            content=processed_content,
            source=data.get('source'),
            publish_date=data.get('publish_date'),
            tags=tags
        )
        return jsonify({"success": True, "id": doc_id})
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
    
    original_content = data['content']
    processed_content = original_content
    
    # 检查content是否包含HTML内容
    if '<' in original_content and '>' in original_content:
        try:
            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(original_content, 'html.parser')
            
            # 获取纯文本内容
            text_content = rag_system.doc_processor.sanitize_html(original_content)
            
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
                            img_text, confidence = rag_system.doc_processor.process_image(img_response.content)
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
                                doc_text = rag_system.doc_processor.process_pdf(doc_response.content)
                                embedded_contents.append(f"【PDF文档内容】: {doc_text}")
                            elif href.endswith('.docx') or href.endswith('.doc'):
                                doc_text = rag_system.doc_processor.process_word_document(doc_response.content)
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
            processed_content = rag_system.doc_processor.sanitize_html(original_content)
    
    try:
        doc_id = rag_system.add_announcement(
            title=data['title'],
            content=processed_content,
            department=data.get('department'),
            publish_date=data.get('publish_date'),
            importance=data.get('importance', 'normal')
        )
        return jsonify({"success": True, "id": doc_id})
    except Exception as e:
        app.logger.error(f"添加公告时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route('/import/csv', methods=['POST'])
def import_csv_endpoint():
    """批量导入CSV接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "请提供CSV文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "请上传CSV文件"}), 400
    
    doc_type = request.form.get('type')
    if doc_type not in ['news', 'announcement']:
        return jsonify({"error": "文档类型必须是'news'或'announcement'"}), 400
    
    try:
        # 保存上传的文件
        temp_path = f"/tmp/{uuid.uuid4()}.csv"
        file.save(temp_path)
        
        # 导入CSV
        result = rag_system.batch_add_from_csv(temp_path, doc_type)
        
        # 删除临时文件
        os.remove(temp_path)
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"导入CSV时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

# 启动函数
if __name__ == "__main__":
    # 如果存在PORT环境变量，使用它，否则使用默认的5000
    port = int(os.environ.get("PORT", 5000))
    # 在开发模式下启用调试，生产环境应使用 gunicorn
    debug = os.environ.get("FLASK_ENV") == "development"
    
    app.run(host="0.0.0.0", port=port, debug=debug)
        