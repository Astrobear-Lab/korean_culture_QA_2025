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
warnings.filterwarnings('ignore')

class PromptingExperiment:
    def __init__(self, model_name="beomi/Kanana-8B"):
        """
        Phase 1: 프롬프팅 실험
        5가지 다른 프롬프트로 Kanana 8B 성능 테스트
        """
        print(f"Loading {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on {self.device}")
        
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
        
        # 1. Baseline: question만
        prompts['baseline'] = f"질문: {question}\n답변:"
        
        # 2. Simple: question_type + question
        prompts['simple'] = f"[{question_type}] {question}\n답변:"
        
        # 3. Rich: 모든 메타데이터 포함
        prompts['rich'] = f"[분류: {category}] [도메인: {domain}] [유형: {question_type}] [주제: {topic_keyword}]\n질문: {question}\n답변:"
        
        # 4. Expert: 전문가 역할 부여
        prompts['expert'] = f"당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요.\n\n질문: {question}\n답변:"
        
        # 5. Format-aware: 답변 형식 명시
        if question_type == "선다형":
            format_instruction = "정답 번호만 입력하세요."
        elif question_type == "단답형":
            format_instruction = "간단하고 정확한 답변을 입력하세요."
        else:  # 서술형
            format_instruction = "자세하고 완전한 답변을 작성하세요."
            
        prompts['format_aware'] = f"[{question_type}] {question}\n\n{format_instruction}\n답변:"
        
        return prompts
    
    def generate_answer(self, prompt, max_length=512):
        """모델로 답변 생성"""
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩 (입력 프롬프트 제거)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # 답변 정리
            answer = generated_text.strip()
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
                
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
                pred_answer = self.generate_answer(prompt)
                
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
                    rouge_scores = self.evaluate_long_answer(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_pred'] = pred_answer
                    sample_results[f'{prompt_name}_rouge1'] = rouge_scores['rouge1']
                    sample_results[f'{prompt_name}_rouge2'] = rouge_scores['rouge2']
                    sample_results[f'{prompt_name}_rougeL'] = rouge_scores['rougeL']
            
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
            
            for prompt_name in ['baseline', 'simple', 'rich', 'expert', 'format_aware']:
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
                    analysis[question_type][prompt_name] = {
                        'rouge1': np.mean(rouge1_scores),
                        'rouge2': np.mean(rouge2_scores),
                        'rougeL': np.mean(rougeL_scores),
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
                    print(f"{prompt_name:15}: ROUGE-1 = {metrics['rouge1']:.3f}, ROUGE-L = {metrics['rougeL']:.3f} (n={metrics['count']})")
        
        # 전체 평균 (가중 평균)
        print(f"\n🏆 종합 순위:")
        print("-" * 30)
        
        prompt_scores = {}
        for prompt_name in ['baseline', 'simple', 'rich', 'expert', 'format_aware']:
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