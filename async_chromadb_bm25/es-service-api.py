from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import json
import os
import re
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ESService")

app = Flask(__name__)

# 从环境变量获取配置
ES_HOST = os.environ.get('ES_HOST', '192.168.222.128')
ES_PORT = os.environ.get('ES_PORT', '9200')
ES_USERNAME = os.environ.get('ES_USERNAME', '')
ES_PASSWORD = os.environ.get('ES_PASSWORD', '')

# 索引名称
NEWS_INDEX = "cooper_news"
NOTICE_INDEX = "cooper_notice"

class ESService:
    """ElasticSearch服务类"""
    
    def __init__(self):
        """初始化ES客户端并创建索引"""
        self.es_client = None
        self._connect_elasticsearch()
        
    def _connect_elasticsearch(self):
        """连接到Elasticsearch服务器"""
        try:
            # 创建ES客户端
            auth = None
            if ES_USERNAME and ES_PASSWORD:
                auth = (ES_USERNAME, ES_PASSWORD)
                
            self.es_client = Elasticsearch(
                hosts=[f"http://{ES_HOST}:{ES_PORT}"],
                basic_auth=auth,
                verify_certs=False,
                timeout=30
            )
            
            # 验证连接
            if self.es_client.ping():
                logger.info("成功连接到Elasticsearch")
                
                # 检查并创建索引
                self._create_es_indexes()
            else:
                logger.warning("无法连接到Elasticsearch")
                self.es_client = None
        except Exception as e:
            logger.error(f"连接Elasticsearch时出错: {str(e)}")
            self.es_client = None
    
    def _create_es_indexes(self):
        """确保所需的ES索引存在，并设置合适的分析器"""
        if not self.es_client:
            return
        
        # 索引映射定义 - 使用IK分词器支持中文
        mappings = {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "source": {
                    "type": "keyword"
                },
                "department": {
                    "type": "keyword"
                },
                "publish_date": {
                    "type": "date",
                    "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"
                },
                "importance": {
                    "type": "keyword"
                },
                "tags": {
                    "type": "keyword"
                },
                "id": {
                    "type": "keyword"
                },
                "base_id": {
                    "type": "keyword"
                }
            }
        }
        
        # 检查并创建索引
        for index_name in [NEWS_INDEX, NOTICE_INDEX]:
            try:
                if not self.es_client.indices.exists(index=index_name):
                    self.es_client.indices.create(
                        index=index_name,
                        body={
                            "mappings": mappings,
                            "settings": {
                                "analysis": {
                                    "analyzer": {
                                        "ik_smart": {
                                            "type": "custom",
                                            "tokenizer": "ik_smart"
                                        },
                                        "ik_max_word": {
                                            "type": "custom",
                                            "tokenizer": "ik_max_word"
                                        }
                                    }
                                }
                            }
                        }
                    )
                    logger.info(f"创建索引 {index_name}")
            except Exception as e:
                logger.error(f"创建索引 {index_name} 时出错: {str(e)}")
    
    def index_news(self, document):
        """
        将新闻文档索引到ES
        
        Args:
            document: 包含新闻内容和元数据的字典
            
        Returns:
            bool: 是否成功索引
        """
        if not self.es_client:
            return False
            
        try:
            doc_id = document.get("id") 
            if not doc_id:
                logger.error("缺少文档ID")
                return False
                
            self.es_client.index(
                index=NEWS_INDEX,
                id=doc_id,
                document=document
            )
            logger.info(f"成功索引新闻，ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"索引新闻时出错: {str(e)}")
            return False
    
    def index_notice(self, document):
        """
        将公告文档索引到ES
        
        Args:
            document: 包含公告内容和元数据的字典
            
        Returns:
            bool: 是否成功索引
        """
        if not self.es_client:
            return False
            
        try:
            doc_id = document.get("id")
            if not doc_id:
                logger.error("缺少文档ID")
                return False
                
            self.es_client.index(
                index=NOTICE_INDEX,
                id=doc_id,
                document=document
            )
            logger.info(f"成功索引公告，ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"索引公告时出错: {str(e)}")
            return False
    
    def search_news(self, query, n_results=5):
        """
        使用BM25搜索新闻
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            list: 新闻结果列表
        """
        if not self.es_client:
            return []
        
        try:
            # 增强查询 - 处理可能的字段名和特殊术语
            enhanced_query = self._enhance_query(query)
            
            # 使用布尔查询，组合多个搜索字段
            response = self.es_client.search(
                index=NEWS_INDEX,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"title": {"query": query, "boost": 2.0}}},
                                {"match": {"content": {"query": query, "boost": 1.0}}},
                                {"match": {"title": {"query": enhanced_query, "boost": 1.5}}},
                                {"match": {"content": {"query": enhanced_query, "boost": 0.8}}},
                                {"match_phrase": {"title": {"query": query, "boost": 3.0, "slop": 2}}},
                                {"match_phrase": {"content": {"query": query, "boost": 1.5, "slop": 4}}}
                            ]
                        }
                    },
                    "size": n_results * 2,  # 请求更多结果，以便过滤重复base_id
                    "_source": True
                }
            )
            
            # 格式化结果
            formatted_results = []
            seen_base_ids = set()
            
            # 提取结果
            hits = response.get("hits", {}).get("hits", [])
            
            for hit in hits:
                source = hit.get("_source", {})
                base_id = source.get("base_id", hit.get("_id", ""))
                
                # 如果已经包含了这个base_id的文档，则跳过
                if base_id in seen_base_ids:
                    continue
                    
                seen_base_ids.add(base_id)
                
                formatted_results.append({
                    "id": base_id,
                    "title": source.get("title", ""),
                    "source": source.get("source", "未知来源"),
                    "publish_date": source.get("publish_date", ""),
                    "content": source.get("content", ""),
                    "relevance_score": hit.get("_score", 0.0),
                    "search_type": "bm25"
                })
                
                # 如果已经收集了足够的不同文档，就停止
                if len(formatted_results) >= n_results:
                    break
            
            return formatted_results
        except Exception as e:
            logger.error(f"搜索新闻时出错: {str(e)}")
            return []
    
    def search_notice(self, query, n_results=5):
        """
        使用BM25搜索公告
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            list: 公告结果列表
        """
        if not self.es_client:
            return []
        
        try:
            # 增强查询 - 处理可能的字段名和特殊术语
            enhanced_query = self._enhance_query(query)
            
            # 使用布尔查询，组合多个搜索字段
            response = self.es_client.search(
                index=NOTICE_INDEX,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"title": {"query": query, "boost": 2.0}}},
                                {"match": {"content": {"query": query, "boost": 1.0}}},
                                {"match": {"title": {"query": enhanced_query, "boost": 1.5}}},
                                {"match": {"content": {"query": enhanced_query, "boost": 0.8}}},
                                {"match_phrase": {"title": {"query": query, "boost": 3.0, "slop": 2}}},
                                {"match_phrase": {"content": {"query": query, "boost": 1.5, "slop": 4}}}
                            ]
                        }
                    },
                    "size": n_results * 2,  # 请求更多结果，以便过滤重复base_id
                    "_source": True
                }
            )
            
            # 格式化结果
            formatted_results = []
            seen_base_ids = set()
            
            # 提取结果
            hits = response.get("hits", {}).get("hits", [])
            
            for hit in hits:
                source = hit.get("_source", {})
                base_id = source.get("base_id", hit.get("_id", ""))
                
                # 如果已经包含了这个base_id的文档，则跳过
                if base_id in seen_base_ids:
                    continue
                    
                seen_base_ids.add(base_id)
                
                formatted_results.append({
                    "id": base_id,
                    "title": source.get("title", ""),
                    "department": source.get("department", "未知部门"),
                    "publish_date": source.get("publish_date", ""),
                    "importance": source.get("importance", "normal"),
                    "content": source.get("content", ""),
                    "relevance_score": hit.get("_score", 0.0),
                    "search_type": "bm25"
                })
                
                # 如果已经收集了足够的不同文档，就停止
                if len(formatted_results) >= n_results:
                    break
            
            return formatted_results
        except Exception as e:
            logger.error(f"搜索公告时出错: {str(e)}")
            return []
    
    def search_all(self, query, n_results=5):
        """
        同时搜索新闻和公告
        
        Args:
            query: 查询文本
            n_results: 每种类型返回的结果数量
            
        Returns:
            dict: 包含新闻和公告结果的字典
        """
        news_results = self.search_news(query, n_results)
        notice_results = self.search_notice(query, n_results)
        
        return {
            "news": news_results,
            "announcements": notice_results
        }
    
    def delete_news(self, doc_id):
        """
        删除新闻
        
        Args:
            doc_id: 文档ID或基础ID
            
        Returns:
            bool: 是否成功删除
        """
        if not self.es_client:
            return False
            
        try:
            if '_' in doc_id:  # 这是一个分块ID
                base_id = doc_id.split('_')[0]
                # 搜索所有具有相同base_id的文档
                response = self.es_client.search(
                    index=NEWS_INDEX,
                    body={
                        "query": {
                            "term": {
                                "base_id": base_id
                            }
                        },
                        "size": 100,
                        "_source": False
                    }
                )
                
                # 提取所有ID
                ids = [hit["_id"] for hit in response.get("hits", {}).get("hits", [])]
                
                # 删除所有匹配的文档
                if ids:
                    for id in ids:
                        self.es_client.delete(index=NEWS_INDEX, id=id)
                    logger.info(f"删除新闻 base_id: {base_id} 的所有 {len(ids)} 个块")
                    return True
                return False
            else:
                # 尝试删除单个文档
                try:
                    self.es_client.delete(index=NEWS_INDEX, id=doc_id)
                    logger.info(f"删除新闻 id: {doc_id}")
                    return True
                except Exception:
                    # 尝试按base_id删除
                    response = self.es_client.search(
                        index=NEWS_INDEX,
                        body={
                            "query": {
                                "term": {
                                    "base_id": doc_id
                                }
                            },
                            "size": 100,
                            "_source": False
                        }
                    )
                    
                    # 提取所有ID
                    ids = [hit["_id"] for hit in response.get("hits", {}).get("hits", [])]
                    
                    # 删除所有匹配的文档
                    if ids:
                        for id in ids:
                            self.es_client.delete(index=NEWS_INDEX, id=id)
                        logger.info(f"删除新闻 base_id: {doc_id} 的所有 {len(ids)} 个块")
                        return True
                    return False
        except Exception as e:
            logger.error(f"删除新闻时出错: {str(e)}")
            return False
    
    def delete_notice(self, doc_id):
        """
        删除公告
        
        Args:
            doc_id: 文档ID或基础ID
            
        Returns:
            bool: 是否成功删除
        """
        if not self.es_client:
            return False
            
        try:
            if '_' in doc_id:  # 这是一个分块ID
                base_id = doc_id.split('_')[0]
                # 搜索所有具有相同base_id的文档
                response = self.es_client.search(
                    index=NOTICE_INDEX,
                    body={
                        "query": {
                            "term": {
                                "base_id": base_id
                            }
                        },
                        "size": 100,
                        "_source": False
                    }
                )
                
                # 提取所有ID
                ids = [hit["_id"] for hit in response.get("hits", {}).get("hits", [])]
                
                # 删除所有匹配的文档
                if ids:
                    for id in ids:
                        self.es_client.delete(index=NOTICE_INDEX, id=id)
                    logger.info(f"删除公告 base_id: {base_id} 的所有 {len(ids)} 个块")
                    return True
                return False
            else:
                # 尝试删除单个文档
                try:
                    self.es_client.delete(index=NOTICE_INDEX, id=doc_id)
                    logger.info(f"删除公告 id: {doc_id}")
                    return True
                except Exception:
                    # 尝试按base_id删除
                    response = self.es_client.search(
                        index=NOTICE_INDEX,
                        body={
                            "query": {
                                "term": {
                                    "base_id": doc_id
                                }
                            },
                            "size": 100,
                            "_source": False
                        }
                    )
                    
                    # 提取所有ID
                    ids = [hit["_id"] for hit in response.get("hits", {}).get("hits", [])]
                    
                    # 删除所有匹配的文档
                    if ids:
                        for id in ids:
                            self.es_client.delete(index=NOTICE_INDEX, id=id)
                        logger.info(f"删除公告 base_id: {doc_id} 的所有 {len(ids)} 个块")
                        return True
                    return False
        except Exception as e:
            logger.error(f"删除公告时出错: {str(e)}")
            return False
    
    def _enhance_query(self, query):
        """
        增强查询，处理可能的字段名和特殊术语
        
        Args:
            query: 原始查询
            
        Returns:
            str: 增强后的查询
        """
        # 1. 检测可能的字段名（使用引号或驼峰/下划线分隔的词）
        potential_fields = re.findall(r'["\']([^"\']+)["\']|([a-z]+[A-Z][a-zA-Z]*)|([a-z_]+)', query)
        
        # 展平结果并过滤空项
        potential_fields = [field for group in potential_fields for field in group if field]
        
        # 2. 为每个潜在字段创建变体
        enhanced_query = query
        for field in potential_fields:
            # 创建常见的字段名变体
            variants = []
            
            # 驼峰转下划线
            if any(c.isupper() for c in field):
                snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in field]).lstrip('_')
                variants.append(snake_case)
            
            # 下划线转驼峰
            if '_' in field:
                camel_case = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(field.split('_')))
                variants.append(camel_case)
            
            # 构造增强后的查询
            for variant in variants:
                if variant not in query:
                    enhanced_query += f" {variant}"
        
        # 3. 提取特殊业务术语
        business_terms = {
            "订单": ["order", "购买", "购物"],
            "客户": ["customer", "用户", "买家"],
            "产品": ["product", "商品", "货物"],
            "支付": ["payment", "付款", "交易"],
            "状态": ["status", "状况", "情况"],
            "取消": ["cancel", "撤销", "终止"],
            "完成": ["complete", "完结", "结束"],
            "收货": ["delivery", "配送", "接收"]
        }
        
        for term, synonyms in business_terms.items():
            if term in query:
                for synonym in synonyms:
                    if synonym not in query:
                        enhanced_query += f" {synonym}"
        
        return enhanced_query

# 创建ES服务实例
es_service = ESService()

# 定义API路由
@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    if es_service.es_client and es_service.es_client.ping():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    else:
        return jsonify({"status": "unhealthy", "timestamp": datetime.now().isoformat()}), 503

@app.route('/index/news', methods=['POST'])
def index_news():
    """
    索引新闻
    
    请求体示例:
    {
        "id": "12345_0",
        "base_id": "12345",
        "title": "新闻标题",
        "content": "新闻内容...",
        "source": "新闻来源",
        "publish_date": "2025-03-16",
        "tags": "标签1,标签2",
        "chunk_index": 0,
        "total_chunks": 3
    }
    """
    try:
        document = request.json
        if not document:
            return jsonify({"error": "请提供文档内容"}), 400
            
        success = es_service.index_news(document)
        if success:
            return jsonify({"success": True, "message": "新闻索引成功"})
        else:
            return jsonify({"error": "索引失败，请检查ES服务"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/index/notice', methods=['POST'])
def index_notice():
    """
    索引公告
    
    请求体示例:
    {
        "id": "67890_0",
        "base_id": "67890",
        "title": "公告标题",
        "content": "公告内容...",
        "department": "发布部门",
        "publish_date": "2025-03-16",
        "importance": "high",
        "chunk_index": 0,
        "total_chunks": 2
    }
    """
    try:
        document = request.json
        if not document:
            return jsonify({"error": "请提供文档内容"}), 400
            
        success = es_service.index_notice(document)
        if success:
            return jsonify({"success": True, "message": "公告索引成功"})
        else:
            return jsonify({"error": "索引失败，请检查ES服务"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search/news', methods=['POST'])
def search_news():
    """
    搜索新闻
    
    请求体示例:
    {
        "query": "搜索关键词",
        "n_results": 5
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "请提供查询内容"}), 400
            
        query = data['query']
        n_results = data.get('n_results', 5)
        
        results = es_service.search_news(query, n_results)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search/notice', methods=['POST'])
def search_notice():
    """
    搜索公告
    
    请求体示例:
    {
        "query": "搜索关键词",
        "n_results": 5
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "请提供查询内容"}), 400
            
        query = data['query']
        n_results = data.get('n_results', 5)
        
        results = es_service.search_notice(query, n_results)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search/all', methods=['POST'])
def search_all():
    """
    同时搜索新闻和公告
    
    请求体示例:
    {
        "query": "搜索关键词",
        "n_results": 5
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "请提供查询内容"}), 400
            
        query = data['query']
        n_results = data.get('n_results', 5)
        
        results = es_service.search_all(query, n_results)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete/news/<doc_id>', methods=['DELETE'])
def delete_news(doc_id):
    """删除新闻"""
    try:
        success = es_service.delete_news(doc_id)
        if success:
            return jsonify({"success": True, "message": f"成功删除新闻ID: {doc_id}"})
        else:
            return jsonify({"error": f"找不到ID为{doc_id}的新闻"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete/notice/<doc_id>', methods=['DELETE'])
def delete_notice(doc_id):
    """删除公告"""
    try:
        success = es_service.delete_notice(doc_id)
        if success:
            return jsonify({"success": True, "message": f"成功删除公告ID: {doc_id}"})
        else:
            return jsonify({"error": f"找不到ID为{doc_id}的公告"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 如果存在PORT环境变量，使用它，否则使用默认的8085
    port = int(os.environ.get("PORT", 8085))
    app.run(host="0.0.0.0", port=port, debug=False)