import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from rouge_score import rouge_scorer
import warnings
from vllm import LLM, SamplingParams
warnings.filterwarnings('ignore')

class PromptingExperiment:
    def __init__(self, model_name="beomi/Kanana-8B"):
        """
        Phase 1: 프롬프팅 실험
        5가지 다른 프롬프트로 Kanana 8B 성능 테스트
        """
        print(f"Loading {model_name}...")

        # vLLM 모델 로드
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",     # 또는 "float16", "auto"
            trust_remote_code=True,
            max_model_len=1300,   # 최대 입력 길이
            gpu_memory_utilization=0.7,  # GPU 메모리 사용률
        )
        
        # Sampling 파라미터 설정
        self.sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.95
        )
        
        
    def load_data(self, data_path="data"):
        """데이터 로드"""
        data_dir = Path(data_path)
        
        with open(data_dir / "train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(data_dir / "dev.json", "r", encoding="utf-8") as f:
            dev_data = json.load(f)
            
        print(f"Loaded {len(train_data)} train samples, {len(dev_data)} dev samples")
        return train_data, dev_data
    
    def create_prompts(self, sample):
        """5가지 프롬프트 생성"""
        question = sample['input']['question']
        question_type = sample['input']['question_type']
        category = sample['input']['category']
        domain = sample['input']['domain']
        topic_keyword = sample['input']['topic_keyword']
        
        prompts = {}
        
        # 0. System prompt: 전문가 역할 부여
        system_prompt = "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요."

        detailed_system_prompt = """당신은 다양한 문화 지식, 관점, 실행에 기반하여 질문에 신뢰도 높고 정확한 답변을 생성하는 한국어 전문가 AI입니다.

사용자가 입력한 다음 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오:
- 질문 유형(question_type): '선다형', '단답형', 또는 '서술형' 중 하나
- 주제(topic_keyword): 문제의 핵심 키워드
- 질문 내용(question): 사용자가 직접 묻는 질문
- 카테고리(category) 및 도메인(domain): 질문이 속한 전반적인 지식 분야

**출력 형식은 다음과 같이 엄격하게 지켜야 합니다:**

1. **선다형 (Multiple Choice)**  
   - 보기 중 정답에 해당하는 **번호만 숫자**로 출력하십시오.

2. **단답형 (Short Answer)**  
   - 5어절 이내의 **간결하고 정확한 명사 또는 구**로 답하십시오.  
   - **동일 의미의 다양한 표현이 존재할 경우**, 이를 **`#` 기호로 구분**하여 나열하십시오.  
     - 예: "탈춤#탈놀이#탈놀음#가면극#산대놀이#야류#오광대"

3. **서술형 (Descriptive Answer)**  
   - 500자 이내로 **신뢰할 수 있고 일관성 있는 문장으로 설명**하십시오.

답변은 질문에만 집중하고, 불필요한 부연 설명이나 반복은 피하십시오."""

        # 1. Baseline: question만
        prompts['baseline'] = {"system_prompt":system_prompt, "user_prompt":f"질문: {question}\n답변:"}
        
        # 2. Simple: question_type + question
        prompts['simple'] = {"system_prompt":system_prompt, "user_prompt":f"[{question_type}] {question}\n답변:"}
        
        # 3. Rich: 모든 메타데이터 포함
        prompts['rich'] = {"system_prompt":system_prompt, "user_prompt":f"분류: {category}\n도메인: {domain}\n주제: {topic_keyword}\n답변 유형: {question_type}\n\n질문: {question}\n답변:"}

        # 4. Expert: 전문가 역할 부여
        prompts['expert'] = {"system_prompt":system_prompt, "user_prompt":f"당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요.\n질문: {question}\n답변:"}

        # 5. Format-aware: 답변 형식 명시
        if question_type == "선다형":
            format_instruction = "보기 중 정답에 해당하는 **번호만 숫자**로 출력하십시오."
        elif question_type == "단답형":
            format_instruction = "5어절 이내의 **간결하고 정확한 명사 또는 구**로 답하십시오. 동일 의미의 다양한 표현이 존재할 경우, 이를 `#` 기호로 구분하여 나열하십시오."
        else:  # 서술형
            format_instruction = "500자 이내로 **신뢰할 수 있고 일관성 있는 문장으로 설명**하십시오."

        prompts['format_aware'] = {"system_prompt":system_prompt, "user_prompt":f"[{question_type}] {question}\n\n{format_instruction}\n답변:"}

        prompts['detailed'] = {"system_prompt":detailed_system_prompt, "user_prompt":f"category: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n질문: {question}\n답변:"}

        return prompts
    
    def generate_answer(self, prompt, max_length=512):
        """모델로 답변 생성"""
        try:
            messages = [
                {"role": "system", "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['user_prompt']}
            ]

            # 생성
            outputs = self.llm.chat(
                messages=messages,
                sampling_params=self.sampling_params
            )

            # 디코딩 (입력 길이 이후만 추출)
            generated_text = outputs[0].outputs[0].text
            
            # 답변 정리
            answer = generated_text.strip()

            # if '\n' in answer:
            #     answer = answer.split('\n')[0].strip()
                
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def evaluate_multiple_choice(self, pred_answer, true_answer):
        """선다형 평가"""
        # 숫자 추출
        pred_nums = re.findall(r'\b[1-5]\b', pred_answer)
        if pred_nums:
            pred = pred_nums[0]
        else:
            pred = pred_answer.strip()
        
        return 1 if pred == true_answer else 0
    
    def evaluate_short_answer(self, pred_answer, true_answer):
        """단답형 평가"""
        pred_clean = re.sub(r'[^\w가-힣]', '', pred_answer.lower())
        true_clean = re.sub(r'[^\w가-힣]', '', true_answer.lower())
        
        # Exact match
        exact_match = 1 if pred_clean == true_clean else 0
        
        # Partial match (포함 관계)
        partial_match = 1 if true_clean in pred_clean or pred_clean in true_clean else 0
        
        return exact_match, partial_match
    
    def evaluate_long_answer(self, pred_answer, true_answer):
        """서술형 평가 (ROUGE 사용)"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        scores = scorer.score(true_answer, pred_answer)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    def calc_BLEU(self, true, pred, apply_avg=True, apply_best=False, use_mecab=True):
        from nltk.translate.bleu_score import sentence_bleu
        from konlpy.tag import Mecab

        tokenizer = Mecab()
        stacked_bleu = []

        if isinstance(true[0], str):
            true = [[t] for t in true]

        for i in range(len(true)):
            best_bleu = 0
            sum_bleu = 0
            for ref_text in true[i]:
                if use_mecab:
                    ref   = tokenizer.morphs(ref_text)
                    candi = tokenizer.morphs(pred[i])
                else:
                    ref   = ref_text.split()
                    candi = pred[i].split()

                score = sentence_bleu([ref], candi, weights=(1, 0, 0, 0))
                sum_bleu   += score
                best_bleu   = max(best_bleu, score)

            avg_bleu = sum_bleu / len(true[i])
            if apply_best:  stacked_bleu.append(best_bleu)
            if apply_avg:   stacked_bleu.append(avg_bleu)

        return sum(stacked_bleu) / len(stacked_bleu)

    def calc_bertscore(self, true, pred):
        import evaluate
        bert_scorer = evaluate.load('bertscore')
        scores = bert_scorer.compute(
            predictions=pred,
            references=true,
            model_type='bert-base-multilingual-cased',
            lang='ko',
            batch_size=1,
        )
        return sum(scores['f1']) / len(scores['f1'])

    # def calc_bleurt(self, true, pred):
    #     from bleurt import score
    #     scorer = score.BleurtScorer('/workspace/korean_culture_QA_2025/bleurt/BLEURT-20')
    #     # BLEURT expects flat lists of strings
    #     flat_true = [t if isinstance(t, str) else t[0] for t in true]
    #     scores = scorer.score(references=flat_true, candidates=pred, batch_size=64)
    #     return sum(scores) / len(scores)

    def run_experiment(self, data, sample_size=None, save_results=True):
        """실험 실행"""
        if sample_size:
            data = data[:sample_size]
            print(f"Using {sample_size} samples for testing")
        
        results = {
            '선다형': [],
            '단답형': [],
            '서술형': []
        }
        
        for i, sample in enumerate(tqdm(data, desc="Running experiments")):
            question_type = sample['input']['question_type']
            true_answer = sample['output']['answer']
            
            # 5가지 프롬프트 생성
            prompts = self.create_prompts(sample)
            
            sample_results = {
                'id': sample['id'],
                'question_type': question_type,
                'true_answer': true_answer,
                'question': sample['input']['question']
            }
            
            # 각 프롬프트로 답변 생성 및 평가
            for prompt_name, prompt in prompts.items():
                pred_answer = self.generate_answer(prompt=prompt)
                
                # 질문 유형별 평가
                if question_type == "선다형":
                    score = self.evaluate_multiple_choice(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_pred'] = pred_answer
                    sample_results[f'{prompt_name}_score'] = score
                    
                elif question_type == "단답형":
                    exact, partial = self.evaluate_short_answer(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_pred'] = pred_answer
                    sample_results[f'{prompt_name}_exact'] = exact
                    sample_results[f'{prompt_name}_partial'] = partial
                    
                else:  # 서술형
                    # 1) Rouge
                    rouge_scores = self.evaluate_long_answer(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_pred'] = pred_answer
                    sample_results[f'{prompt_name}_rouge1'] = rouge_scores['rouge1']
                    sample_results[f'{prompt_name}_rouge2'] = rouge_scores['rouge2']
                    sample_results[f'{prompt_name}_rougeL'] = rouge_scores['rougeL']

                    # 2) BLEU
                    bleu_score = self.calc_BLEU([true_answer], [pred_answer])
                    sample_results[f'{prompt_name}_bleu'] = bleu_score

                    # 3) BERTScore 
                    bertscore = self.calc_bertscore(
                        [true_answer],
                        [pred_answer],
                    )
                    sample_results[f'{prompt_name}_bertscore'] = bertscore

                    # 4) BLEURT
                    # bleurt_score = self.calc_bleurt(
                    #     [true_answer],
                    #     [pred_answer],
                    # )
                    # sample_results[f'{prompt_name}_bleurt'] = bleurt_score
            
            results[question_type].append(sample_results)
            
            # 중간 저장 (10개마다)
            if (i + 1) % 10 == 0 and save_results:
                self.save_intermediate_results(results, i + 1)
        
        return results
    
    def save_intermediate_results(self, results, current_idx):
        """중간 결과 저장"""
        save_path = f"phase1_intermediate_results_{current_idx}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def analyze_results(self, results):
        """결과 분석 및 요약"""
        analysis = {}
        
        for question_type in ['선다형', '단답형', '서술형']:
            if not results[question_type]:
                continue
                
            type_data = results[question_type]
            analysis[question_type] = {}
            
            for prompt_name in ['baseline', 'simple', 'rich', 'expert', 'format_aware', 'detailed']:
                if question_type == "선다형":
                    scores = [item[f'{prompt_name}_score'] for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'accuracy': np.mean(scores),
                        'count': len(scores)
                    }
                    
                elif question_type == "단답형":
                    exact_scores = [item[f'{prompt_name}_exact'] for item in type_data]
                    partial_scores = [item[f'{prompt_name}_partial'] for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'exact_match': np.mean(exact_scores),
                        'partial_match': np.mean(partial_scores),
                        'count': len(exact_scores)
                    }
                    
                else:  # 서술형
                    rouge1_scores = [item[f'{prompt_name}_rouge1'] for item in type_data]
                    rouge2_scores = [item[f'{prompt_name}_rouge2'] for item in type_data]
                    rougeL_scores = [item[f'{prompt_name}_rougeL'] for item in type_data]
                    bleu_scores     = [item[f'{prompt_name}_bleu']      for item in type_data]
                    bert_scores     = [item[f'{prompt_name}_bertscore'] for item in type_data]
                    # bleurt_scores   = [item[f'{prompt_name}_bleurt']   for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'rouge1': np.mean(rouge1_scores),
                        'rouge2': np.mean(rouge2_scores),
                        'rougeL': np.mean(rougeL_scores),
                        'bleu': np.mean(bleu_scores),
                        'bertscore': np.mean(bert_scores),
                        # 'bleurt': np.mean(bleurt_scores),
                        'count': len(rouge1_scores)
                    }
        
        return analysis
    
    def print_analysis(self, analysis):
        """분석 결과 출력"""
        print("\n" + "="*80)
        print("PHASE 1: PROMPTING EXPERIMENT RESULTS")
        print("="*80)
        
        for question_type, type_results in analysis.items():
            print(f"\n📊 {question_type} 결과:")
            print("-" * 50)
            
            if question_type == "선다형":
                for prompt_name, metrics in type_results.items():
                    print(f"{prompt_name:15}: Accuracy = {metrics['accuracy']:.3f} (n={metrics['count']})")
                    
            elif question_type == "단답형":
                for prompt_name, metrics in type_results.items():
                    print(f"{prompt_name:15}: Exact = {metrics['exact_match']:.3f}, Partial = {metrics['partial_match']:.3f} (n={metrics['count']})")
                    
            else:  # 서술형
                for prompt_name, metrics in type_results.items():
                    print(
                        f"{prompt_name:15}: "
                        f"ROUGE-1 = {metrics['rouge1']:.3f}, "
                        f"ROUGE-2 = {metrics['rouge2']:.3f}, "
                        f"ROUGE-L = {metrics['rougeL']:.3f}, "
                        f"BLEU = {metrics['bleu']:.3f}, "
                        f"BERTScore = {metrics['bertscore']:.3f}, "
                        # f"BLEURT = {metrics['bleurt']:.3f} "
                        f"(n={metrics['count']})"
                    )
        
        # 전체 평균 (가중 평균)
        print(f"\n🏆 종합 순위:")
        print("-" * 30)
        
        prompt_scores = {}
        for prompt_name in ['baseline', 'simple', 'rich', 'expert', 'format_aware', 'detailed']:
            total_score = 0
            total_count = 0
            
            for question_type, type_results in analysis.items():
                if prompt_name in type_results:
                    if question_type == "선다형":
                        score = type_results[prompt_name]['accuracy']
                    elif question_type == "단답형":
                        score = type_results[prompt_name]['exact_match']
                    else:  # 서술형
                        score = type_results[prompt_name]['rouge1']
                    
                    count = type_results[prompt_name]['count']
                    total_score += score * count
                    total_count += count
            
            if total_count > 0:
                prompt_scores[prompt_name] = total_score / total_count
        
        # 순위 정렬
        ranked_prompts = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (prompt_name, score) in enumerate(ranked_prompts, 1):
            print(f"{i}. {prompt_name:15}: {score:.3f}")
    
    def save_final_results(self, results, analysis):
        """최종 결과 저장"""
        # 상세 결과
        with open('phase1_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 분석 요약
        with open('phase1_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장 완료:")
        print(f"   - phase1_detailed_results.json")
        print(f"   - phase1_analysis_summary.json")

def main():
    """메인 실행 함수"""
    print("🚀 Phase 1: Prompting Experiment with Kanana 8B")
    print("="*60)
    
    # 실험 초기화
    experiment = PromptingExperiment("beomi/Kanana-8B")
    
    # 데이터 로드
    train_data, dev_data = experiment.load_data()
    
    # Dev set으로 실험 (시간 절약을 위해)
    print(f"\n🔬 Dev set으로 실험 시작 (n={len(dev_data)})")
    
    # 실험 실행
    results = experiment.run_experiment(dev_data, sample_size=None)
    
    # 결과 분석
    analysis = experiment.analyze_results(results)
    
    # 결과 출력
    experiment.print_analysis(analysis)
    
    # 결과 저장
    experiment.save_final_results(results, analysis)
    
    print(f"\n✅ Phase 1 실험 완료!")

if __name__ == "__main__":
    main() 