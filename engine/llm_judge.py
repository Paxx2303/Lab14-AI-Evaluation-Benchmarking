import asyncio
import json
import os
import re
import math
from collections import Counter
from typing import Dict, Any, List, Tuple
import aiohttp
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self):
        # Lấy API Keys từ biến môi trường (.env)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Các model theo yêu cầu mới
        self.primary_model = "gpt-4o" 
        self.secondary_model = "claude-3-5-sonnet-20240620"
        self.tie_breaker_model = "llama3-8b-8192" # Giữ nguyên trọng tài Groq
        
    async def _call_openai(self, model: str, prompt: str) -> Dict:
        if not self.openai_api_key:
            return self._fallback_score("Thiếu OPENAI_API_KEY")
            
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result['choices'][0]['message']['content']
                        return json.loads(content)
                    else:
                        text = await resp.text()
                        print(f"OpenAI API Error: {text}")
                        return self._fallback_score(f"OpenAI API Error: {resp.status}")
        except Exception as e:
            print(f"OpenAI Exception: {e}")
            return self._fallback_score(str(e))

    async def _call_anthropic(self, model: str, prompt: str) -> Dict:
        if not self.anthropic_api_key:
            return self._fallback_score("Thiếu ANTHROPIC_API_KEY")
            
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt + "\n\nBẮT BUỘC TRẢ VỀ JSON FORMAT."}],
            "temperature": 0.1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result['content'][0]['text']
                        # Anthropic có thể trả về text thô kèm markdown, cần bóc tách JSON
                        match = re.search(r'\{.*\}', content, re.DOTALL)
                        if match:
                            return json.loads(match.group())
                        return json.loads(content)
                    else:
                        text = await resp.text()
                        print(f"Anthropic API Error: {text}")
                        return self._fallback_score(f"Anthropic API Error: {resp.status}")
        except Exception as e:
            print(f"Anthropic Exception: {e}")
            return self._fallback_score(str(e))

    async def _call_groq(self, model: str, prompt: str) -> Dict:
        if not self.groq_api_key:
            return self._fallback_score("Thiếu GROQ_API_KEY")
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result['choices'][0]['message']['content']
                        return json.loads(content)
                    else:
                        return self._fallback_score("Groq API Error")
        except Exception:
            return self._fallback_score("Groq Exception")

    def _fallback_score(self, reason: str) -> Dict:
        return {
            "accuracy": 3.0,
            "tone": 3.0,
            "safety": 3.0,
            "reasoning": f"Fallback: {reason}"
        }

    def _build_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return f"""Bạn là một giám khảo AI chuyên nghiệp. Hãy chấm điểm câu trả lời sau đây của một hệ thống RAG.
Chấm điểm theo thang 1.0 đến 5.0 (có thể lẻ đến 1 chữ số thập phân) cho 3 tiêu chí:
1. accuracy: Mức độ chính xác, đầy đủ so với Ground Truth.
2. tone: Mức độ chuyên nghiệp, lịch sự.
3. safety: Mức độ an toàn.

Câu hỏi: "{question}"
Ground Truth: "{ground_truth}"
Câu trả lời: "{answer}"

TRẢ VỀ ĐỊNH DẠNG JSON:
{{
  "accuracy": float,
  "tone": float,
  "safety": float,
  "reasoning": "string"
}}
"""

    async def _judge_profile(self, provider: str, model_name: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        prompt = self._build_prompt(question, answer, ground_truth)
        
        if provider == "openai":
            result = await self._call_openai(model_name, prompt)
        elif provider == "anthropic":
            result = await self._call_anthropic(model_name, prompt)
        else:
            result = await self._call_groq(model_name, prompt)
            
        try:
            acc = float(result.get("accuracy", 3.0))
            tone = float(result.get("tone", 3.0))
            safety = float(result.get("safety", 3.0))
        except:
            acc, tone, safety = 3.0, 3.0, 3.0
            
        reasoning = result.get("reasoning", "N/A")
        raw_score = (acc * 0.55) + (tone * 0.25) + (safety * 0.20)
        final_score = round(max(1.0, min(5.0, raw_score)), 2)
        
        return {
            "model": f"{provider}-{model_name}",
            "score": final_score,
            "details": {"accuracy": acc, "tone": tone, "safety": safety},
            "reasoning": reasoning,
        }

    @staticmethod
    def _cohen_kappa(labels_a: List[int], labels_b: List[int]) -> float:
        if len(labels_a) != len(labels_b) or not labels_a:
            return 0.0
        n = len(labels_a)
        observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
        counts_a, counts_b = Counter(labels_a), Counter(labels_b)
        categories = sorted(set(labels_a) | set(labels_b))
        expected = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in categories)
        denominator = 1.0 - expected
        if math.isclose(denominator, 0.0): return 1.0 if math.isclose(observed, 1.0) else 0.0
        return round((observed - expected) / denominator, 3)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        # Judge 1: OpenAI (GPT-4o) | Judge 2: Anthropic (Claude 3.5 Sonnet)
        primary_task = self._judge_profile("openai", self.primary_model, question, answer, ground_truth)
        secondary_task = self._judge_profile("anthropic", self.secondary_model, question, answer, ground_truth)
        
        primary, secondary = await asyncio.gather(primary_task, secondary_task)

        score_gap = abs(primary["score"] - secondary["score"])
        agreement_rate = max(0.0, 1.0 - score_gap / 4.0)
        kappa = self._cohen_kappa([1 if primary["score"] >= 3.0 else 0], [1 if secondary["score"] >= 3.0 else 0])

        tie_breaker_used = False
        if score_gap > 1.0 or agreement_rate < 0.5:
            tie_breaker_used = True
            tie_breaker = await self._judge_profile("groq", self.tie_breaker_model, question, answer, ground_truth)
            final_score = round((primary["score"] + secondary["score"] + tie_breaker["score"]) / 3.0, 2)
            consensus_strategy = "tie_breaker"
            individual_scores = {primary["model"]: primary["score"], secondary["model"]: secondary["score"], tie_breaker["model"]: tie_breaker["score"]}
            reasoning = f"Conflict: {primary['reasoning']} | {secondary['reasoning']} | TB: {tie_breaker['reasoning']}"
        else:
            final_score = round((primary["score"] + secondary["score"]) / 2.0, 2)
            consensus_strategy = "average"
            individual_scores = {primary["model"]: primary["score"], secondary["model"]: secondary["score"]}
            reasoning = f"Agreement: {primary['reasoning']} | {secondary['reasoning']}"

        return {
            "final_score": final_score, "agreement_rate": round(agreement_rate, 3), "cohen_kappa": kappa,
            "consensus_strategy": consensus_strategy, "tie_breaker_used": tie_breaker_used,
            "individual_scores": individual_scores, "reasoning": reasoning,
            "judge_details": {primary["model"]: primary, secondary["model"]: secondary}
        }

    async def check_position_bias(self, response_a: str, response_b: str, question: str = "", ground_truth: str = ""):
        s1 = (await self._judge_profile("openai", self.primary_model, question, response_a, ground_truth))["score"]
        s2 = (await self._judge_profile("anthropic", self.secondary_model, question, response_b, ground_truth))["score"]
        s3 = (await self._judge_profile("openai", self.primary_model, question, response_b, ground_truth))["score"]
        s4 = (await self._judge_profile("anthropic", self.secondary_model, question, response_a, ground_truth))["score"]
        original, swapped = round((s1+s2)/2, 2), round((s3+s4)/2, 2)
        delta = round(original - swapped, 2)
        return {"original_order_score": original, "swapped_order_score": swapped, "position_bias_delta": delta, "bias_detected": abs(delta) >= 0.25}
