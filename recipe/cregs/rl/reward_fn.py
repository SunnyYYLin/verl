import asyncio
import math

import httpx


async def fetch_score(client: httpx.AsyncClient, url: str, text: str) -> float:
    """异步获取单个文本的分数。"""
    try:
        response = await client.post(url, json={'text': text})
        response.raise_for_status()
        scores = response.json().get('score', [])
        if isinstance(scores, list) and len(scores) > 0:
            return sum(scores) / len(scores)
        return float(scores) if scores else 0.0
    except Exception:
        return 0.0


async def get_all_scores_async(url: str, texts: list[str]) -> list[float]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_score(client, url, t) for t in texts]
        return await asyncio.gather(*tasks)


def compute_score_gene_specificity(
    solution_str: str,
    ground_truth: int,
    extra_info: dict,
    **kwargs,
) -> float:
    # 构造正样本和负样本序列
    positive = extra_info.get("gene", "") + solution_str
    negatives = [neg + solution_str for neg in extra_info.get("negatives", [])]
    
    all_texts = [positive] + negatives
    server_url = extra_info.get("reward_server_url", "http://localhost:8080/score")

    # 异步并行请求分数
    all_scores = asyncio.run(get_all_scores_async(server_url, all_texts))
    
    # 将分数视为 logits，计算 Log-Softmax 作为奖励 (等价于负交叉熵 -CE)
    # Reward = log(exp(pos_score) / sum(exp(all_scores)))
    max_score = max(all_scores)
    log_sum_exp = max_score + math.log(sum(math.exp(s - max_score) for s in all_scores))
    specificity = pos_score - log_sum_exp

    return specificity
