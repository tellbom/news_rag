# rerank_service.py
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import os
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class RerankService:
    def __init__(self, model_path="/models/bge-reranker-base"):
        """初始化重排序服务"""
        self.device = "cpu"
        app.logger.info(f"使用设备: {self.device}")
        
        # 记录启动时间
        start_time = time.time()
        
        # 加载模型和分词器
        app.logger.info(f"正在加载重排序模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # 针对CPU优化的模型加载
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            local_files_only=True,
            torchscript=True  # 启用TorchScript优化
        )
        self.model.eval()
        
        # 报告加载时间
        load_time = time.time() - start_time
        app.logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 设置批处理大小 - CPU通常使用较小的批量
        self.batch_size = int(os.environ.get("RERANK_BATCH_SIZE", "4"))
        app.logger.info(f"批处理大小: {self.batch_size}")
    
    def rerank(self, query, documents, top_k=None):
        """对文档进行重排序"""
        if not documents:
            return []
            
        # 准备batch输入
        features = []
        for doc in documents:
            # 获取文档内容 - 依据输入格式进行适配
            if isinstance(doc, dict):
                content = doc.get("content", "") or doc.get("text", "")
                title = doc.get("title", "")
                # 组合标题和内容
                if title:
                    text = f"{title}\n{content}"
                else:
                    text = content
            else:
                text = str(doc)
            
            # 编码查询和文档对
            features.append({
                "text_pair": (query, text),
                "original_doc": doc  # 保留原始文档
            })
        
        # 批处理
        batch_size = int(os.environ.get("RERANK_BATCH_SIZE", "4"))
        all_scores = []
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            
            # 生成输入
            encoded = self.tokenizer(
                [pair["text_pair"][0] for pair in batch],
                [pair["text_pair"][1] for pair in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            # 计算分数
            try:
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    
                    # 正确处理输出，适应不同版本的模型
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = getattr(outputs, 'logits', outputs)
                    
                    # 检查logits的形状并添加日志
                    app.logger.info(f"Logits shape: {logits.shape}")
                    
                    # 根据logits的形状修改获取分数的方式
                    if logits.shape[1] == 1:
                        # 如果只有一个输出维度（回归分数）
                        scores = logits.squeeze(-1).cpu().numpy()
                    else:
                        # 如果是多类（通常是二分类），取第二类（正类）的分数
                        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                # 将原始文档与分数对应
                for j, score in enumerate(scores):
                    all_scores.append((batch[j]["original_doc"], float(score)))
                    
            except Exception as e:
                app.logger.info(f"处理批次 {i//batch_size+1} 时出错: {str(e)}")
                import traceback
                app.logger.info(traceback.format_exc())
                # 继续处理其他批次
                continue
        
        # 如果没有成功处理的结果，返回原始文档
        if not all_scores:
            app.logger.info("无法计算任何重排序分数，返回原始文档")
            return documents
        
        # 排序
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 限制返回数量
        if top_k is not None:
            all_scores = all_scores[:top_k]
        
        # 添加得分并返回
        reranked = []
        for doc, score in all_scores:
            # 如果是字典，直接添加得分
            if isinstance(doc, dict):
                doc_with_score = doc.copy()
                doc_with_score["rerank_score"] = score
                reranked.append(doc_with_score)
            else:
                # 如果不是字典，转换为字典
                reranked.append({"content": str(doc), "rerank_score": score})
        
        return reranked

# 初始化服务
rerank_service = RerankService()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/rerank', methods=['POST'])
def rerank():
    data = request.json
    
    if not data:
        return jsonify({"error": "请提供数据"}), 400
        
    query = data.get("query")
    documents = data.get("documents", [])
    top_k = data.get("top_k")
    
    if not query:
        return jsonify({"error": "请提供查询内容"}), 400
        
    try:
        reranked_docs = rerank_service.rerank(query, documents, top_k)
        return jsonify({"results": reranked_docs})
    except Exception as e:
        app.logger.info(f"重排序失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091)