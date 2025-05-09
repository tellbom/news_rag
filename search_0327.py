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


class VectorServiceClient:
    """向量服务客户端，用于与Qdrant+LlamaIndex服务通信"""

    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        初始化向量服务客户端

        Args:
            base_url: 向量服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.is_available = self._check_health()
        self.session = requests.Session()
        self.logger = logging.getLogger("VectorServiceClient")

    def _check_health(self) -> bool:
        """
        检查向量服务是否可用

        Returns:
            bool: 服务是否可用
        """
        logger = logging.getLogger("VectorServiceClient")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception as e:
            self.logger.error(f"向量服务健康检查失败: {str(e)}")
            return False

    def search_vector(self, query: str, n_results: int = 5, search_type: str = "all") -> Dict:
        """
        执行向量搜索

        Args:
            query: 查询文本
            n_results: 返回结果数量
            search_type: 搜索类型 (all, news, announcements)

        Returns:
            Dict: 搜索结果
        """
        if not self.is_available:
            return {"news": [], "announcements": []}

        try:
            response = self.session.post(
                f"{self.base_url}/vector/search",
                json={
                    "query": query,
                    "n_results": n_results,
                    "search_type": search_type,
                    "mmr_enabled": True
                },
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"向量搜索失败: {response.text}")
                return {"news": [], "announcements": []}
        except Exception as e:
            self.logger.error(f"向量搜索请求出错: {str(e)}")
            return {"news": [], "announcements": []}

    def add_news(self, title: str, content: str, source: str = None,
                 publish_date: str = None, tags: List = None, id: str = None) -> Dict:
        """
        添加新闻到向量服务

        Args:
            title: 新闻标题
            content: 新闻内容（已处理的纯文本）
            source: 来源
            publish_date: 发布日期
            tags: 标签列表
            id: 文档ID

        Returns:
            Dict: 包含任务ID的响应
        """
        if not self.is_available:
            return {"success": False, "error": "向量服务不可用"}

        data = {
            "title": title,
            "content": content
        }

        if source:
            data["source"] = source
        if publish_date:
            data["publish_date"] = publish_date
        if tags:
            data["tags"] = tags
        if id:
            data["id"] = id

        try:
            response = self.session.post(
                f"{self.base_url}/news",
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"添加新闻失败: {response.text}")
                return {"success": False, "error": response.text}
        except Exception as e:
            self.logger.error(f"添加新闻请求出错: {str(e)}")
            return {"success": False, "error": str(e)}

    def add_announcement(self, title: str, content: str, department: str = None,
                         publish_date: str = None, importance: str = "normal", id: str = None) -> Dict:
        """
        添加公告到向量服务

        Args:
            title: 公告标题
            content: 公告内容（已处理的纯文本）
            department: 部门
            publish_date: 发布日期
            importance: 重要性
            id: 文档ID

        Returns:
            Dict: 包含任务ID的响应
        """
        if not self.is_available:
            return {"success": False, "error": "向量服务不可用"}

        data = {
            "title": title,
            "content": content
        }

        if department:
            data["department"] = department
        if publish_date:
            data["publish_date"] = publish_date
        if importance:
            data["importance"] = importance
        if id:
            data["id"] = id

        try:
            response = self.session.post(
                f"{self.base_url}/announcement",
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"添加公告失败: {response.text}")
                return {"success": False, "error": response.text}
        except Exception as e:
            self.logger.error(f"添加公告请求出错: {str(e)}")
            return {"success": False, "error": str(e)}

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

        try:
            response = self.session.delete(
                f"{self.base_url}/news/{doc_id}",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"删除新闻请求出错: {str(e)}")
            return False

    def delete_announcement(self, doc_id: str) -> bool:
        """
        删除公告

        Args:
            doc_id: 文档ID

        Returns:
            bool: 是否成功删除
        """
        if not self.is_available:
            return False

        try:
            response = self.session.delete(
                f"{self.base_url}/announcement/{doc_id}",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"删除公告请求出错: {str(e)}")
            return False

    def get_task_status(self, task_id: str) -> Dict:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict: 任务状态
        """
        if not self.is_available:
            return {"status": "unavailable"}

        try:
            response = self.session.get(
                f"{self.base_url}/task/{task_id}",
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"获取任务状态失败: {response.text}")
                return {"status": "error", "error": response.text}
        except Exception as e:
            self.logger.error(f"获取任务状态请求出错: {str(e)}")
            return {"status": "error", "error": str(e)}

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
            doc_id: 文档ID或基础ID

        Returns:
            bool: 是否成功删除
        """
        success = True

        # 如果启用了混合搜索，从ES删除
        if self.es_client and self.es_client.is_available:
            es_success = self.es_client.delete_news(doc_id)
            if not es_success:
                app.logger.error(f"从ES服务删除新闻 {doc_id} 失败")
                success = False

        return success

    def delete_notice(self, doc_id: str) -> bool:
        """
        删除公告

        Args:
            doc_id: 文档ID或基础ID

        Returns:
            bool: 是否成功删除
        """
        success = True
        # 如果启用了混合搜索，从ES删除
        if self.es_client and self.es_client.is_available:
            es_success = self.es_client.delete_notice(doc_id)
            if not es_success:
                app.logger.error(f"从ES服务删除公告 {doc_id} 失败")
                success = False

        return success

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

# RankFusion类算法类
class RankFusion:
    """结果融合算法工具类，用于融合向量、BM25和重排序结果"""

    @staticmethod
    def contextual_fusion(query: str, dense_results: dict, lexical_results: dict, k: int = 60,
                          rerank_url: str = None, rerank_weight: float = 0.4,
                          vector_weight: float = None, lexical_weight: float = None) -> dict:
        """
        上下文感知的融合算法，针对不同类型的查询动态调整权重，并整合重排序结果

        Args:
            query: 用户查询
            dense_results: 向量检索结果 (格式: {"news": [...], "announcements": [...]})
            lexical_results: 文本检索结果 (格式: {"news": [...], "announcements": [...]})
            k: RRF常数
            rerank_url: 重排序服务URL，如果提供则使用重排序服务
            rerank_weight: 重排序结果的权重，范围0-1
            vector_weight: 向量搜索权重(可选)，如果提供则覆盖自动计算的权重
            lexical_weight: 文本搜索权重(可选)，如果提供则覆盖自动计算的权重

        Returns:
            融合后的结果字典 (格式: {"news": [...], "announcements": [...]})
        """
        # 用于保存融合结果
        fused_results = {"news": [], "announcements": []}

        # 如果没有提供外部权重，则分析查询特征确定权重
        if vector_weight is None or lexical_weight is None:
            # 提取查询特征
            query_terms = set(query.lower().split())
            is_status_query = any(term in query_terms for term in ['状态', '取消', '完成', '支付'])
            is_time_query = any(term in query_terms for term in ['时间', '日期', '年', '月', '日'])
            is_type_query = any(term in query_terms for term in ['类型', '种类', '分类'])

            # 动态调整权重
            if is_status_query or is_type_query:
                v_weight = 0.4
                l_weight = 0.6
            elif is_time_query:
                v_weight = 0.5
                l_weight = 0.5
            else:
                v_weight = 0.7
                l_weight = 0.3

            # 使用计算得到的权重
            vector_weight = vector_weight if vector_weight is not None else v_weight
            lexical_weight = lexical_weight if lexical_weight is not None else l_weight

        # 记录使用的权重（可用于调试）
        app.logger.info(f"融合使用权重: vector={vector_weight}, lexical={lexical_weight}")

        # 处理新闻和公告
        for content_type in ["news", "announcements"]:
            # 获取各自的结果
            vector_content = dense_results.get(content_type, [])
            lexical_content = lexical_results.get(content_type, [])

            # 合并结果，建立完整的文档集合
            all_items = {}  # id -> item
            rerank_candidates = []  # 用于提交给重排序

            # 收集向量结果
            for item in vector_content:
                item_id = item.get("id", "")
                if not item_id:
                    continue

                # 确保源类型标记正确
                item["source_type"] = item.get("source_type", "vector")
                all_items[item_id] = item

                # 添加到重排序候选集
                if "content" not in item and "text" in item:
                    item_copy = item.copy()
                    item_copy["content"] = item_copy.get("text", "")
                    rerank_candidates.append(item_copy)
                else:
                    rerank_candidates.append(item.copy())

            # 收集BM25结果
            for item in lexical_content:
                item_id = item.get("id", "")
                if not item_id:
                    continue

                # 确保源类型标记正确
                item["source_type"] = item.get("source_type", "lexical")

                if item_id not in all_items:
                    all_items[item_id] = item
                    # 添加到重排序候选集
                    if "content" not in item and "text" in item:
                        item_copy = item.copy()
                        item_copy["content"] = item_copy.get("text", "")
                        rerank_candidates.append(item_copy)
                    else:
                        rerank_candidates.append(item.copy())

            # 初始化分数字典
            scores = {item_id: {"item": item, "score": 0.0, "matches": set()}
                      for item_id, item in all_items.items()}

            # 重排序处理 - 获取重排序分数但不直接使用结果
            rerank_scores = {}  # id -> score
            if rerank_url and rerank_candidates:
                try:
                    # 调用重排序服务
                    response = requests.post(
                        rerank_url + "/rerank",
                        json={
                            "query": query,
                            "documents": rerank_candidates,
                            "top_k": len(rerank_candidates)  # 保留所有结果
                        },
                        timeout=10
                    )

                    if response.status_code == 200:
                        # 提取重排序分数
                        reranked = response.json().get("results", [])

                        # 计算归一化的重排序分数 (倒数排名)
                        for rank, item in enumerate(reranked, start=1):
                            item_id = item.get("id", "")
                            if item_id:
                                # 使用RRF公式计算重排序分数
                                rerank_scores[item_id] = 1.0 / (k + rank)

                                # 如果有相关性分数，也考虑进去
                                if "score" in item:
                                    rerank_scores[item_id] *= (1 + 0.2 * item["score"])
                except Exception as e:
                    app.logger.error(f"调用重排序服务失败: {str(e)}")
                    # 失败时rerank_scores保持为空

            # 计算向量检索分数
            for rank, item in enumerate(vector_content, start=1):
                item_id = item.get("id", "")
                if not item_id or item_id not in scores:
                    continue

                # 获取相关性分数 - 兼容不同格式
                relevance_score = item.get("relevance_score", 0.0)

                # 加权RRF分数，考虑原始相关性分数
                scores[item_id]["score"] += vector_weight * (1.0 / (k + rank)) * (1 + 0.2 * relevance_score)
                scores[item_id]["matches"].add("vector")

            # 计算BM25检索分数
            for rank, item in enumerate(lexical_content, start=1):
                item_id = item.get("id", "")
                if not item_id or item_id not in scores:
                    continue

                # 基础RRF分数
                scores[item_id]["score"] += lexical_weight * (1.0 / (k + rank))
                scores[item_id]["matches"].add("lexical")

                # 额外的词匹配奖励
                content = item.get("content", "")
                title = item.get("title", "")

                # 检查结果是否包含查询词
                term_matches = sum(1 for term in query_terms if term in (content + title).lower())
                term_match_ratio = term_matches / len(query_terms) if query_terms else 0

                # 词匹配奖励
                scores[item_id]["score"] *= (1 + 0.2 * term_match_ratio)

            # 整合重排序分数
            if rerank_scores:
                for item_id in scores:
                    if item_id in rerank_scores:
                        # 使用配置的权重添加重排序分数
                        scores[item_id]["score"] += rerank_weight * rerank_scores[item_id]
                        scores[item_id]["matches"].add("rerank")

            # 多检索源奖励
            for item_id, data in scores.items():
                # 计算匹配源的数量
                match_count = len(data["matches"])

                # 根据匹配源数量给予不同程度的奖励
                if match_count == 3:  # 同时在向量、BM25和重排序中出现
                    data["score"] *= 1.4
                elif match_count == 2:  # 在两种检索中出现
                    data["score"] *= 1.25

            # 按分数排序
            sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

            # 提取排序后的结果项
            fused_results[content_type] = [item_data["item"] for item_data in sorted_items]

            # 添加调试信息（可选）
            for i, item in enumerate(fused_results[content_type]):
                item_id = item.get("id", "")
                if item_id in scores:
                    item["fusion_score"] = scores[item_id]["score"]
                    item["fusion_sources"] = list(scores[item_id]["matches"])

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

    def submit_task(self, task_func: Callable, callback: Optional[Callable] = None, *args, **kwargs) -> str:
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

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
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

    def clean_old_tasks(self, max_age: float = 3600):
        """
        清理旧任务数据

        Args:
            max_age: 最大保留时间（秒），默认1小时
        """
        # 实现清理逻辑...
        pass


class DocumentClient:
    """文档处理客户端，用于与文档处理微服务通信"""

    def __init__(self, base_url: str = "http://localhost:5002"):
        """
        初始化文档处理客户端

        Args:
            base_url: 文档处理微服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.is_available = self._check_health()
        self.session = requests.Session()
        self.logger = logging.getLogger("DocumentClient")

    def _check_health(self) -> bool:
        """
        检查文档处理服务是否可用

        Returns:
            bool: 服务是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception as e:
            self.logger.error(f"文档处理服务健康检查失败: {str(e)}")
            return False

    def process_image(self, image_url: str) -> Tuple[str, float]:
        """
        从图像URL中提取文本

        Args:
            image_url: 图像URL

        Returns:
            Tuple[str, float]: 提取的文本和置信度
        """
        if not self.is_available:
            return "文档处理服务不可用", 0.0

        try:
            response = self.session.post(
                f"{self.base_url}/process/image",
                json={"url": image_url},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", ""), result.get("metadata", {}).get("confidence", 0.0)
            else:
                self.logger.error(f"处理图像失败: {response.text}")
                return f"处理图像失败: {response.text}", 0.0
        except Exception as e:
            self.logger.error(f"处理图像请求出错: {str(e)}")
            return f"处理图像请求出错: {str(e)}", 0.0

    def process_word_document(self, doc_url: str) -> str:
        """
        从Word文档URL中提取文本

        Args:
            doc_url: Word文档URL

        Returns:
            str: 提取的文本
        """
        if not self.is_available:
            return "文档处理服务不可用"

        try:
            response = self.session.post(
                f"{self.base_url}/process/word",
                json={"url": doc_url},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                self.logger.error(f"处理Word文档失败: {response.text}")
                return f"处理Word文档失败: {response.text}"
        except Exception as e:
            self.logger.error(f"处理Word文档请求出错: {str(e)}")
            return f"处理Word文档请求出错: {str(e)}"

    def process_pdf(self, pdf_url: str) -> str:
        """
        从PDF URL中提取文本

        Args:
            pdf_url: PDF URL

        Returns:
            str: 提取的文本
        """
        if not self.is_available:
            return "文档处理服务不可用"

        try:
            response = self.session.post(
                f"{self.base_url}/process/pdf",
                json={"url": pdf_url},
                timeout=60  # PDF处理可能需要更长时间
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                self.logger.error(f"处理PDF失败: {response.text}")
                return f"处理PDF失败: {response.text}"
        except Exception as e:
            self.logger.error(f"处理PDF请求出错: {str(e)}")
            return f"处理PDF请求出错: {str(e)}"

    def process_ppt(self, ppt_url: str) -> str:
        """
        从PPT URL中提取文本

        Args:
            ppt_url: PPT URL

        Returns:
            str: 提取的文本
        """
        if not self.is_available:
            return "文档处理服务不可用"

        try:
            response = self.session.post(
                f"{self.base_url}/process/ppt",
                json={"url": ppt_url},
                timeout=60  # PPT处理可能需要更长时间
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                self.logger.error(f"处理PPT失败: {response.text}")
                return f"处理PPT失败: {response.text}"
        except Exception as e:
            self.logger.error(f"处理PPT请求出错: {str(e)}")
            return f"处理PPT请求出错: {str(e)}"

    def sanitize_html(self, html_content: str) -> str:
        """
        清理HTML内容

        Args:
            html_content: HTML内容

        Returns:
            str: 清理后的文本
        """
        if not self.is_available:
            # 如果服务不可用，使用本地备用方法
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return '\n'.join(chunk for chunk in chunks if chunk)
            except Exception as e:
                self.logger.error(f"本地HTML清理错误: {str(e)}")
                return html_content

        try:
            response = self.session.post(
                f"{self.base_url}/process/html",
                json={"html": html_content},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                self.logger.error(f"处理HTML失败: {response.text}")
                # 回退到本地处理
                return self.sanitize_html_local(html_content)
        except Exception as e:
            self.logger.error(f"处理HTML请求出错: {str(e)}")
            # 回退到本地处理
            return self.sanitize_html_local(html_content)

    def sanitize_html_local(self, html_content: str) -> str:
        """
        本地备用的HTML清理方法，当服务不可用时使用

        Args:
            html_content: HTML内容

        Returns:
            str: 清理后的文本
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.extract()

            # 获取文本
            text = soup.get_text(separator=' ', strip=True)

            # 清理空白区域
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            self.logger.error(f"本地HTML清理错误: {str(e)}")
            # 如果清理失败，返回原始内容
            return html_content


class ChineseRAGSystem:
    def __init__(
            self,
            llm_api_key: str = None,
            llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            llm_model: str = "qwen-plus",
            use_langchain: bool = True,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            use_hybrid_search: bool = False,
            es_service_url: str = "http://localhost:8085",
            vector_service_url: str = "http://localhost:5001",
            document_service_url: str = "http://localhost:5002"
    ):
        """
        初始化中文RAG系统

        Args:
            llm_api_key: 大模型API密钥
            llm_base_url: 大模型API基础URL
            llm_model: 使用的大模型名称
            use_langchain: 是否使用LangChain进行文档分块
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠部分大小
            use_hybrid_search: 是否开启混合检索(vector+bm25)
            es_service_url: es服务器
            vector_service_url: 向量服务URL
        """

        # ES服务配置
        self.use_hybrid_search = use_hybrid_search
        self.es_service_url = es_service_url

        # 初始化文档处理客户端
        self.document_service_url = document_service_url
        self.doc_client = DocumentClient(base_url=document_service_url)
        app.logger.info(f"文档处理客户端初始化完成，服务可用: {self.doc_client.is_available}")

        # 是否使用LangChain
        self.use_langchain = use_langchain
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化向量服务客户端
        self.vector_service_url = vector_service_url
        self.vector_client = VectorServiceClient(base_url=vector_service_url)
        app.logger.info(f"向量服务客户端初始化完成，服务可用: {self.vector_client.is_available}")

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


        # 初始化LangChain文本分割器
        if self.use_langchain:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )

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
                    {"role": "user", "content": user_prompt}  # 用户角色提示
                ],
                temperature=temperature,  # 控制多样性
                max_tokens=max_tokens  # 控制回答长度
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
                context.append(f"{i + 1}. 标题: {news['title']}")
                context.append(f"   来源: {news['source']} ({news['publish_date']})")
                context.append(f"   内容: {news['content']}")
                context.append("")

        # 添加公告
        if search_results["announcements"]:
            context.append("## 相关公告")
            for i, announcement in enumerate(search_results["announcements"]):
                importance_marker = "🔴" if announcement['importance'] == "high" else "🟢"
                context.append(f"{i + 1}. {importance_marker} {announcement['title']}")
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
            use_hybrid_search: 是否使用混合搜索（向量+BM25+rerank倒排），None表示使用配置值

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
                app.logger.info(f"使用混合搜索（向量+BM25+rerank倒排）检索结果")
            else:
                search_results = self.search_vector(query, n_results)
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
                "search_type": "hybrid" if use_hybrid_search else "vector"
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
                # 检查content是否包含HTML内容
                if '<' in original_content and '>' in original_content:
                    try:
                        # 使用文档处理服务清理HTML
                        text_content = self.doc_client.sanitize_html(original_content)

                        # 处理嵌入的图片
                        embedded_contents = []
                        soup = BeautifulSoup(original_content, 'html.parser')
                        for img in soup.find_all('img'):
                            src = img.get('src', '')
                            if src and (src.startswith('http://') or src.startswith('https://')):
                                try:
                                    # 使用文档处理服务处理图片
                                    img_text, confidence = self.doc_client.process_image(src)
                                    if img_text and img_text != "No text detected in image" and img_text != "图像中未检测到文本":
                                        embedded_contents.append(f"【图片内容】: {img_text}")
                                except Exception as e:
                                    app.logger.warning(f"处理嵌入图片时出错: {str(e)}")

                        # 处理嵌入的文档链接
                        for a in soup.find_all('a'):
                            href = a.get('href', '')
                            if href:
                                try:
                                    if href.endswith('.pdf'):
                                        # 使用文档处理服务处理PDF
                                        doc_text = self.doc_client.process_pdf(href)
                                        embedded_contents.append(f"【PDF文档内容】: {doc_text}")
                                    elif href.endswith('.docx') or href.endswith('.doc'):
                                        # 使用文档处理服务处理Word
                                        doc_text = self.doc_client.process_word_document(href)
                                        embedded_contents.append(f"【Word文档内容】: {doc_text}")
                                    elif href.endswith('.pptx') or href.endswith('.ppt'):
                                        # 使用文档处理服务处理PPT
                                        doc_text = self.doc_client.process_ppt(href)
                                        embedded_contents.append(f"【PPT文档内容】: {doc_text}")
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
                        processed_content = self.doc_client.sanitize_html(original_content)

                # 1. 向量服务：发送到向量服务
                if self.vector_client and self.vector_client.is_available:
                    vector_response = self.vector_client.add_news(
                        title=title,
                        content=processed_content,
                        source=source,
                        publish_date=publish_date,
                        tags=tags,
                        id=base_id
                    )

                    if vector_response.get("success"):
                        app.logger.info(f"成功将新闻提交到向量服务，任务ID: {vector_response.get('task_id')}")
                    else:
                        app.logger.error(f"提交新闻到向量服务失败: {vector_response.get('error')}")

                # 2. ES服务：如果启用了混合搜索，同步到ES
                if self.es_client and self.es_client.is_available:
                    # 分割长文本
                    document_chunks = self.split_text(processed_content)
                    app.logger.info(f"将新闻文章 '{title}' 分割为 {len(document_chunks)} 个块")

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

                        # 异步索引到ES
                        self.es_client.index_news(es_document, async_mode=True)

                    app.logger.info(f"已将新闻同步索引到ES服务")

                return base_id

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
                        # 使用文档处理服务清理HTML
                        text_content = self.doc_client.sanitize_html(original_content)

                        # 处理嵌入的图片
                        embedded_contents = []
                        soup = BeautifulSoup(original_content, 'html.parser')
                        for img in soup.find_all('img'):
                            src = img.get('src', '')
                            if src and (src.startswith('http://') or src.startswith('https://')):
                                try:
                                    # 使用文档处理服务处理图片
                                    img_text, confidence = self.doc_client.process_image(src)
                                    if img_text and img_text != "No text detected in image" and img_text != "图像中未检测到文本":
                                        embedded_contents.append(f"【图片内容】: {img_text}")
                                except Exception as e:
                                    app.logger.warning(f"处理嵌入图片时出错: {str(e)}")

                        # 处理嵌入的文档链接
                        for a in soup.find_all('a'):
                            href = a.get('href', '')
                            if href:
                                try:
                                    if href.endswith('.pdf'):
                                        # 使用文档处理服务处理PDF
                                        doc_text = self.doc_client.process_pdf(href)
                                        embedded_contents.append(f"【PDF文档内容】: {doc_text}")
                                    elif href.endswith('.docx') or href.endswith('.doc'):
                                        # 使用文档处理服务处理Word
                                        doc_text = self.doc_client.process_word_document(href)
                                        embedded_contents.append(f"【Word文档内容】: {doc_text}")
                                    elif href.endswith('.pptx') or href.endswith('.ppt'):
                                        # 使用文档处理服务处理PPT
                                        doc_text = self.doc_client.process_ppt(href)
                                        embedded_contents.append(f"【PPT文档内容】: {doc_text}")
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
                        processed_content = self.doc_client.sanitize_html(original_content)

                # 1. 向量服务：发送到向量服务
                if self.vector_client and self.vector_client.is_available:
                    vector_response = self.vector_client.add_announcement(
                        title=title,
                        content=processed_content,
                        department=department,
                        publish_date=publish_date,
                        importance=importance,
                        id=base_id
                    )

                    if vector_response.get("success"):
                        app.logger.info(f"成功将公告提交到向量服务，任务ID: {vector_response.get('task_id')}")
                    else:
                        app.logger.error(f"提交公告到向量服务失败: {vector_response.get('error')}")

                # 2. ES服务：如果启用了混合搜索，同步到ES
                if self.es_client and self.es_client.is_available:
                    # 分割长文本
                    document_chunks = self.split_text(processed_content)
                    app.logger.info(f"将公告 '{title}' 分割为 {len(document_chunks)} 个块")

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

                        # 异步索引到ES
                        self.es_client.index_notice(es_document, async_mode=True)

                    app.logger.info(f"已将公告同步索引到ES服务")

                return base_id

            except Exception as e:
                app.logger.error(f"异步添加公告时出错: {str(e)}")
                raise

        # 提交到任务管理器
        task_id = task_manager.submit_task(process_and_add_task)
        app.logger.info(f"提交添加公告任务: {task_id}, 文档基础ID: {base_id}")

        return task_id

    def search_vector(self, query: str, n_results: int = 5, search_type: str = "all") -> Dict[str, List[Dict]]:
        """
        使用向量服务进行搜索

        Args:
            query: 查询文本
            n_results: 返回结果数量
            search_type: 搜索类型 (all, news, announcements)

        Returns:
            Dict[str, List[Dict]]: 搜索结果
        """
        if not self.vector_client or not self.vector_client.is_available:
            app.logger.warning("向量服务不可用，无法执行向量搜索")
            return {"news": [], "announcements": []}

        # 调用向量服务客户端
        return self.vector_client.search_vector(query, n_results, search_type)

    def hybrid_search_all(self, query: str, n_results: int = 5, vector_weight: float = None,
                          lexical_weight: float = None) -> Dict[str, List[Dict]]:
        """
        混合搜索：使用向量检索和BM25融合搜索结果

        Args:
            query: 查询文本
            n_results: 每种类型返回的结果数量
            vector_weight: 向量搜索权重(可选)，如果提供则覆盖默认值
            lexical_weight: 文本搜索权重(可选)，如果提供则覆盖默认值

        Returns:
            Dict[str, List[Dict]]: 融合后的搜索结果
        """
        # 1. 向量检索 - 使用向量服务
        vector_results = self.search_vector(query, n_results)

        # 2. 如果ES服务可用，执行BM25检索
        if self.es_client and self.es_client.is_available:
            bm25_results = self.es_client.search_all(query, n_results)

            # 3. 融合结果
            try:
                # 获取重排序服务URL
                rerank_url = RERANK_SERVICE_URL if 'RERANK_SERVICE_URL' in globals() else None

                # 使用融合方法，传入权重配置
                fused_results = RankFusion.contextual_fusion(
                    query=query,
                    dense_results=vector_results,
                    lexical_results=bm25_results,
                    rerank_url=rerank_url,
                    vector_weight=vector_weight,
                    lexical_weight=lexical_weight
                )

                app.logger.info(f"使用{'重排序增强的' if rerank_url else ''}混合搜索结果")
                return fused_results
            except Exception as e:
                app.logger.error(f"融合搜索结果时出错: {str(e)}")
                # 出错时回退到向量检索结果
                return vector_results

        # 如果ES服务不可用，直接返回向量检索结果
        return vector_results

    def delete_news(self, doc_id: str) -> bool:
        success = True
        # 1. 从向量服务删除
        if self.vector_client and self.vector_client.is_available:
            vector_success = self.vector_client.delete_news(doc_id)
            if not vector_success:
                success = False
        # 2. 从ES删除
        if self.es_client and self.es_client.is_available:
            es_success = self.es_client.delete_news(doc_id)
            if not es_success:
                success = False
        return success

    def delete_announcement(self, doc_id: str) -> bool:
        success = True
        # 1. 从向量服务删除
        if self.vector_client and self.vector_client.is_available:
            vector_success = self.vector_client.delete_announcement(doc_id)
            if not vector_success:
                success = False
        # 2. 从ES删除
        if self.es_client and self.es_client.is_available:
            es_success = self.es_client.delete_notice(doc_id)
            if not es_success:
                success = False
        return success

# 初始化RAG系统
# 从环境变量获取配置
VECTOR_SERVICE_URL = os.environ.get('VECTOR_SERVICE_URL', 'http://192.168.140.100:5001')
LLM_API_KEY = os.environ.get('LLM_API_KEY', 'sk-9f8124e18aa242af830c8a502c015c40')
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen-plus')
CHROMA_PORT = os.environ.get('CHROMA_PORT', '8000')
USE_LANGCHAIN = os.environ.get('USE_LANGCHAIN', 'true').lower() == 'true'
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))
USE_HYBRID_SEARCH = os.environ.get('use_hybrid_search', 'true').lower() == 'true'
ES_SERVICE_URL = os.environ.get('es_service_url', 'http://192.168.140.100:8085')
RERANK_SERVICE_URL = os.environ.get('rerank_service_url', 'http://192.168.140.100:8091')
DOCUMENT_SERVICE_URL = os.environ.get('document_service_url', 'http://localhost:5002')  # 新增

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
            llm_api_key=LLM_API_KEY,
            llm_base_url=LLM_BASE_URL,
            llm_model=LLM_MODEL,
            use_langchain=USE_LANGCHAIN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_hybrid_search=USE_HYBRID_SEARCH,
            es_service_url=ES_SERVICE_URL if USE_HYBRID_SEARCH else None,
            vector_service_url=VECTOR_SERVICE_URL,
            document_service_url=DOCUMENT_SERVICE_URL  # 添加文档服务URL
        )

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
            "vector_service_available": rag_system.vector_client.is_available if rag_system else False,
            "es_service_available": rag_system.es_client.is_available if rag_system and rag_system.es_client else False
        }
    })


@app.route('/vector_status', methods=['GET'])
def vector_status_endpoint():
    """向量服务状态接口"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500

    if rag_system.vector_client:
        is_available = rag_system.vector_client.is_available
        return jsonify({
            "vector_service_enabled": True,
            "vector_service_available": is_available,
            "vector_service_url": rag_system.vector_service_url
        })
    else:
        return jsonify({
            "vector_service_enabled": False,
            "vector_service_available": False
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
        success = rag_system.delete_news(doc_id)
        if success:
            return jsonify({"success": True, "message": f"成功删除新闻: {doc_id}"})
        else:
            return jsonify({"error": f"找不到ID为{doc_id}的新闻"}), 404
    except Exception as e:
        app.logger.error(f"删除新闻时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete/announcement/<doc_id>', methods=['DELETE'])
def delete_announcement_endpoint(doc_id):
    """删除指定ID的公告"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500

    try:
        success = rag_system.delete_announcement(doc_id)
        if success:
            return jsonify({"success": True, "message": f"成功删除公告: {doc_id}"})
        else:
            return jsonify({"error": f"找不到ID为{doc_id}的新闻"}), 404
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
        # 保存文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            file.save(temp.name)
            temp_path = temp.name

        # 构建URL
        file_type = file.content_type or os.path.splitext(file.filename)[1]

        # 使用本地文件系统路径 - 在实际环境中应替换为可访问的URL
        # 这里假设我们在同一环境中，可以共享文件路径
        # 实际应用中可能需要上传到对象存储并获取URL

        if file_type.startswith('image/') or file_type.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            result_json = requests.post(
                f"{rag_system.document_service_url}/process/image",
                json={"url": temp_path},  # 这里需要实际系统支持
                timeout=30
            ).json()
        elif file_type.startswith('application/pdf') or file_type.endswith('.pdf'):
            result_json = requests.post(
                f"{rag_system.document_service_url}/process/pdf",
                json={"url": temp_path},
                timeout=60
            ).json()
        elif (file_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml.document') or
              file_type.startswith('application/msword') or
              file_type.endswith(('.doc', '.docx'))):
            result_json = requests.post(
                f"{rag_system.document_service_url}/process/word",
                json={"url": temp_path},
                timeout=30
            ).json()
        elif (file_type.startswith('application/vnd.openxmlformats-officedocument.presentationml.presentation') or
              file_type.startswith('application/vnd.ms-powerpoint') or
              file_type.endswith(('.ppt', '.pptx'))):
            result_json = requests.post(
                f"{rag_system.document_service_url}/process/ppt",
                json={"url": temp_path},
                timeout=60
            ).json()
        else:
            # 对于不支持的文件类型，返回错误
            os.unlink(temp_path)  # 清理临时文件
            return jsonify({"error": f"不支持的文件类型: {file_type}"}), 400

        # 清理临时文件
        os.unlink(temp_path)

        # 添加文件信息
        result_json["metadata"]["filename"] = file.filename
        result_json["metadata"]["original_file_type"] = file_type

        return jsonify(result_json)
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
        sanitized_content = rag_system.doc_client.sanitize_html(html_content)

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
    """混合查询接口（向量+BM25+rerank倒排排序）"""
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


#专门用于获取提示词和检索内容
@app.route('/extract_context', methods=['POST'])
def extract_context_endpoint():
    """提取上下文和生成提示词接口，不调用LLM"""
    if not rag_system:
        return jsonify({"error": "RAG系统尚未初始化"}), 500

    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "请提供查询内容"}), 400

    query = data['query']
    n_results = data.get('n_results', 3)

    # 接收LLM返回的权重配置
    weights_config = data.get('weights_config', None)
    search_metadata = {}

    # 解析权重配置
    if weights_config:
        try:
            # 如果是字符串(JSON文本)，尝试解析成字典
            if isinstance(weights_config, str):
                weights_config = json.loads(weights_config)

            vector_weight = float(weights_config.get('vector_weight', 0.7))
            lexical_weight = float(weights_config.get('lexical_weight', 0.3))
            query_type = weights_config.get('query_type', 'general')
            reasoning = weights_config.get('reasoning', '')

            # 记录使用的权重配置
            search_metadata = {
                "vector_weight": vector_weight,
                "lexical_weight": lexical_weight,
                "query_type": query_type,
                "reasoning": reasoning,
                "source": "llm_analysis"
            }

            app.logger.info(
                f"使用LLM分析的查询权重: vector={vector_weight}, lexical={lexical_weight}, type={query_type}")
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            app.logger.error(f"解析权重配置时出错: {str(e)}")
            # 使用默认值
            vector_weight = None
            lexical_weight = None
            search_metadata = {"error": f"权重配置解析失败: {str(e)}"}
    else:
        # 未提供权重配置，使用系统默认值
        vector_weight = None
        lexical_weight = None
        search_metadata = {"source": "default_weights"}

    try:
        # 确定是否使用混合搜索
        use_hybrid_search = data.get('use_hybrid_search')
        if use_hybrid_search is None:
            use_hybrid_search = rag_system.use_hybrid_search and rag_system.es_client and rag_system.es_client.is_available

        # 执行检索
        if use_hybrid_search and rag_system.es_client and rag_system.es_client.is_available:
            # 使用传入的权重配置调用混合搜索
            search_results = rag_system.hybrid_search_all(
                query=query,
                n_results=n_results,
                vector_weight=vector_weight,
                lexical_weight=lexical_weight
            )
        else:
            search_results = rag_system.search_vector(query, n_results)

        # 格式化上下文
        context = rag_system.format_context(search_results)

        # 构建系统提示
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

        # 构建用户提示
        user_prompt = f"""用户问题: {query}

        ----上下文信息----
        {context}
        ----上下文信息结束----

        基于上述上下文信息，请回答用户的问题。如果上下文信息不足以回答用户问题，请明确指出。"""
        # 返回提示词和上下文
        return jsonify({
            "success": True,
            "query": query,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": context,
            "search_results": search_results,
            "search_type": "hybrid" if use_hybrid_search else "vector",
            "search_metadata": search_metadata
        })
    except Exception as e:
        app.logger.error(f"提取上下文和提示词时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

# LLM上下文感知
@app.route('/classify_query', methods=['POST'])
def classify_query_endpoint():
    """提供查询分类所需的提示词和内容"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "请提供查询内容"}), 400

    query = data['query']

    # 构建系统提示词
    system_prompt = """
    你是一个专业的搜索查询分析器。你的任务是分析用户的查询意图，并确定在混合检索系统中使用的最佳权重。

    请分析查询的语义和意图，然后确定它属于以下哪种类型：
    1. 事实型查询：寻找具体事实、状态、数字或确切信息的查询
    2. 时间型查询：与日期、时间、时间段相关的查询
    3. 分类型查询：寻找特定类别、种类或分类体系的查询
    4. 概念型查询：寻找解释、定义或概念性内容的查询
    5. 通用型查询：不属于以上类别的一般性查询

    对于不同类型的查询，应分配不同的检索权重：
    - 事实型查询：词汇匹配(0.6)更重要，语义匹配(0.4)次之
    - 时间型查询：词汇匹配(0.5)和语义匹配(0.5)同等重要
    - 分类型查询：词汇匹配(0.6)更重要，语义匹配(0.4)次之
    - 概念型查询：语义匹配(0.8)更重要，词汇匹配(0.2)次之
    - 通用型查询：语义匹配(0.7)更重要，词汇匹配(0.3)次之

    请分析查询并返回JSON格式的结果，包含以下字段：
    - query_type: 查询类型(factual, temporal, categorical, conceptual, general)
    - vector_weight: 语义匹配权重(0.0-1.0)
    - lexical_weight: 词汇匹配权重(0.0-1.0)
    - reasoning: 你的分析理由(简短说明)
    """

    # 构建用户提示词
    user_prompt = f"""
    请分析以下用户查询，确定查询类型并推荐合适的检索权重：

    用户查询: "{query}"

    请以JSON格式返回结果。只返回JSON对象，不要有其他说明文字。
    """

    # 返回提示词和查询
    return jsonify({
        "success": True,
        "query": query,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    })

# 启动函数
if __name__ == "__main__":
    # 如果存在PORT环境变量，使用它，否则使用默认的5000
    port = int(os.environ.get("PORT", 5000))
    # 在开发模式下启用调试，生产环境应使用 gunicorn
    debug = os.environ.get("FLASK_ENV") == "development"

    app.run(host="0.0.0.0", port=port, debug=debug)