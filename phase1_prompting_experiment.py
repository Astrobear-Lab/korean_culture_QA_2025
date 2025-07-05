from __init__ import *

class PromptingExperiment:
    def __init__(self, model_name="beomi/Kanana-8B", load_model: bool = True):
        """
        Phase 1: 프롬프팅 실험
        5가지 다른 프롬프트로 Kanana 8B 성능 테스트
        """
        print(f"Loading {model_name}...")
        self.model_name = model_name

        self.llm = None
        self.sampling_params = None
        if load_model:
            print(f"Loading {model_name}...")
            self.llm = LLM(
                model=self.model_name,
                dtype="bfloat16",     # 또는 "float16", "auto"
                trust_remote_code=True,
                max_model_len=1300,   # 최대 입력 길이
                gpu_memory_utilization=0.9,  # GPU 메모리 사용률
            )

            # Sampling 파라미터 설정 (모델 로드 여부와 상관없이 준비)
            self.sampling_params = SamplingParams(
                max_tokens=2048,
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
        with open(data_dir / "test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(train_data)} train, {len(dev_data)} dev" + (f", {len(test_data)} test" if test_data else ""))
        return train_data, dev_data, test_data
    
    def create_prompts(self, sample):
        """5가지 프롬프트 생성"""
        question = sample['input']['question']
        question_type = sample['input']['question_type']
        category = sample['input']['category']
        domain = sample['input']['domain']
        topic_keyword = sample['input']['topic_keyword']
        
        prompts = {}
        
        # 0. System prompt: 전문가 역할 부여
        system_prompt = """당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요.
당신의 답변은 다음과 같은 형식을 따라야 합니다:
1. **선다형 (Multiple Choice)**  
   - 보기 중 정답에 해당하는 번호만 **숫자**로 출력하십시오.

2. **단답형 (Short Answer)**  
   - 5어절 이내의 **명사 또는 구**로 답하십시오.  

3. **서술형 (Descriptive Answer)**  
   - 500자 이내의 문장으로 설명하십시오."""

        detailed_system_prompt = """당신은 한국의 문화에 기반하여 질문에 신뢰도 높고 정확한 답변을 생성하는 한국어 전문가 AI입니다.

사용자가 입력한 다음 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오:
- 질문 유형(question_type): '선다형', '단답형', 또는 '서술형' 중 하나
- 주제(topic_keyword): 문제의 핵심 키워드
- 질문 내용(question): 사용자가 직접 묻는 질문
- 카테고리(category) 및 도메인(domain): 질문이 속한 전반적인 지식 분야

당신의 답변은 다음과 같은 형식을 따라야 합니다:
1. **선다형 (Multiple Choice)**  
   - 보기 중 정답에 해당하는 번호만 **숫자**로 출력하십시오.

2. **단답형 (Short Answer)**  
   - 5어절 이내의 **명사 또는 구**로 답하십시오.  

3. **서술형 (Descriptive Answer)**  
   - 500자 이내의 문장으로 설명하십시오."""

        # 1. Baseline: question만
        # prompts['baseline'] = {"system_prompt":system_prompt, "user_prompt":f"주어진 질문에 적절한 답변을 해주세요.\n질문: {question}\n답변:"}
        
        # 2. Simple: question_type + question
        # prompts['simple'] = {"system_prompt":system_prompt, "user_prompt":f"주어진 질문에 적절한 답변을 해주세요.\n<{question_type}>\n<질문>\n{question}\n답변:"}
        
        # 3. Rich: 모든 메타데이터 포함
        prompts['rich'] = {"system_prompt":system_prompt, "user_prompt":f"주어진 질문에 적절한 답변을 해주세요.\n\n분류: {category}\n도메인: {domain}\n주제: {topic_keyword}\n답변 유형: {question_type}\n\n<질문>\n{question}\n\n답변:"}

        # 4. Expert: 전문가 역할 부여
        # prompts['expert'] = {"system_prompt":system_prompt, "user_prompt":f"당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요.\n질문: {question}\n답변:"}

        # 5. Format-aware: 답변 형식 명시
        if question_type == "선다형":
            format_instruction = "보기 중 정답에 해당하는 번호만 **숫자**로 출력하십시오."
        elif question_type == "단답형":
            format_instruction = "5어절 이내의 **명사 또는 구**로 답하십시오."
        else:  # 서술형
            format_instruction = "500자 이내의 문장으로 설명하십시오."

        prompts['format_aware'] = {"system_prompt":system_prompt, "user_prompt":f"주어진 질문에 적절한 답변을 해주세요.\n\n<{question_type}>\n{format_instruction}\n\n<질문>\n{question}\n\n답변:"}

        prompts['detailed'] = {"system_prompt":detailed_system_prompt, "user_prompt":f"주어진 질문에 적절한 답변을 해주세요.\n\ncategory: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n<질문>\n{question}\n\n답변:"}

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
        # """단답형 평가"""
        # pred_clean = re.sub(r'[^\w가-힣]', '', pred_answer.lower())
        # true_clean = re.sub(r'[^\w가-힣]', '', true_answer.lower())
        
        # # Exact match
        # exact_match = 1 if pred_clean == true_clean else 0
        
        # # Partial match (포함 관계)
        # partial_match = 1 if true_clean in pred_clean or pred_clean in true_clean else 0
        
        # return exact_match, partial_match

        """
        Calculate Exact Match score where true_data may contain multiple acceptable answers separated by #
        """
        correct = 0
        true_answer_list = true_answer.split('#')

        if any(pred_answer.strip() == ans.strip() for ans in true_answer_list):
            correct = 1
                
        return correct
    
    def evaluate_long_answer(self, pred_answer, true_answer):
        """서술형 평가 (ROUGE-1, ROUGE-2, ROUGE-L via rouge_metric)"""
        from rouge_metric import Rouge

        rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,        # F1 score
            weight_factor=1.0,
        )

        # 단일 문장 pair도 리스트로 감싸서 전달
        scores = rouge_evaluator.get_scores(
            [pred_answer],
            [true_answer]
        )

        return {
            'rouge1':   scores['rouge-1']['f'],
            'rouge2':   scores['rouge-2']['f'],
            'rougeL':   scores['rouge-l']['f']
        }

    def calc_BLEU(self, pred_answer: str, true_answer: str, apply_avg=True, apply_best=False, use_mecab=True):
        from nltk.translate.bleu_score import sentence_bleu
        from konlpy.tag import Mecab

        tokenizer = Mecab()
        stacked_bleu = []

        # 1) 단일 문자열을 참조 리스트로 감싸기
        refs = [true_answer]  # 리스트 of reference strings
        cand = pred_answer    # 하나의 candidate string

        # 2) 토크나이징
        if use_mecab:
            cand_tokens = tokenizer.morphs(cand)
        else:
            cand_tokens = cand.split()

        # 3) 각 ref 텍스트별로 BLEU 계산
        best_bleu = 0
        sum_bleu  = 0
        for ref_text in refs:
            if use_mecab:
                ref_tokens = tokenizer.morphs(ref_text)
            else:
                ref_tokens = ref_text.split()

            score = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))
            sum_bleu  += score
            best_bleu = max(best_bleu, score)

        # 4) 평균
        avg_bleu = sum_bleu / len(refs)  # 이제 len(refs) == 1

        # 5) 결과 수집
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)

        return sum(stacked_bleu) / len(stacked_bleu) if stacked_bleu else 0.0


    # def calc_bertscore(self, true, pred):
    #     import evaluate
    #     bert_scorer = evaluate.load('bertscore')
    #     scores = bert_scorer.compute(
    #         predictions=pred,
    #         references=true,
    #         model_type='bert-base-multilingual-cased',
    #         lang='ko',
    #         batch_size=1,
    #     )
    #     return sum(scores['f1']) / len(scores['f1'])

    # def calc_bleurt(self, true, pred):
    #     from bleurt import score
    #     scorer = score.BleurtScorer('/workspace/korean_culture_QA_2025/bleurt/BLEURT-20')
    #     # BLEURT expects flat lists of strings
    #     flat_true = [t if isinstance(t, str) else t[0] for t in true]
    #     scores = scorer.score(references=flat_true, candidates=pred, batch_size=64)
    #     return sum(scores) / len(scores)

    def run_experiment(self, data, sample_size=None, save_results=True, test_mode=False):
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
            true_answer = sample['output']['answer'] if 'output' in sample else None
            
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
                sample_results[f'{prompt_name}_pred'] = pred_answer
                if '</think>' in pred_answer:
                    pred_answer = pred_answer.split('</think>')[-1].strip()
                
                if test_mode:
                    continue  # 평가 스킵
                # 질문 유형별 평가
                if question_type == "선다형":
                    score = self.evaluate_multiple_choice(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_score'] = score
                    
                elif question_type == "단답형":
                    exact = self.evaluate_short_answer(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_exact'] = exact
                    # sample_results[f'{prompt_name}_partial'] = partial
                    
                else:  # 서술형
                    # 1) Rouge
                    rouge_scores = self.evaluate_long_answer(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_rouge1'] = rouge_scores['rouge1']
                    sample_results[f'{prompt_name}_rouge2'] = rouge_scores['rouge2']
                    sample_results[f'{prompt_name}_rougeL'] = rouge_scores['rougeL']

                    # 2) BLEU
                    bleu_score = self.calc_BLEU(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_bleu'] = bleu_score

                    # 3) BERTScore 
                    # bertscore = self.calc_bertscore(
                    #     [true_answer],
                    #     [pred_answer],
                    # )
                    # sample_results[f'{prompt_name}_bertscore'] = bertscore

                    # 4) BLEURT
                    # bleurt_score = self.calc_bleurt(
                    #     [true_answer],
                    #     [pred_answer],
                    # )
                    # sample_results[f'{prompt_name}_bleurt'] = bleurt_score
            
            results[question_type].append(sample_results)
            
            # 중간 저장 (10개마다)
            if ((i + 1) % 10 == 0) or (i + 1 == len(data)) and save_results:
                self.save_intermediate_results(results, i + 1)
        
        return results
    
    def save_intermediate_results(self, results, current_idx):
        """중간 결과 저장"""
        save_path = f"results/phase1_{self.model_name.split('/')[-1]}_intermediate_results_{current_idx}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def only_scoring(self, data: dict) -> dict:
        """
        data: {
        "선다형": [ { "id":..., "question_type":"선다형", "true_answer":..., "question":..., 
                    "rich_pred":..., "format_aware_pred":..., ... }, … ],
        "단답형": [ … ],
        "서술형": [ … ]
        }
        """
        results = {
            '선다형': [],
            '단답형': [],
            '서술형': []
        }

        for q_type, samples in data.items():
            for sample in samples:
                rec = sample.copy()  # 원본 레코드 복사

                # `_pred` 로 끝나는 필드만 찾아서 평가
                for key, pred in sample.items():
                    if not key.endswith('_pred'):
                        continue

                    name = key[:-5]  # 'rich_pred' -> 'rich'
                    if '</think>' in pred:
                        pred = pred.split('</think>')[-1].strip()

                    # 질문 유형별 점수 계산
                    if q_type == "선다형":
                        rec[f'{name}_score'] = self.evaluate_multiple_choice(pred, rec['true_answer'])
                    elif q_type == "단답형":
                        rec[f'{name}_exact'] = self.evaluate_short_answer(pred, rec['true_answer'])
                    else:  # 서술형
                        rouge = self.evaluate_long_answer(pred, rec['true_answer'])
                        rec[f'{name}_rouge1'] = rouge['rouge1']
                        rec[f'{name}_rouge2'] = rouge['rouge2']
                        rec[f'{name}_rougeL'] = rouge['rougeL']
                        rec[f'{name}_bleu']   = self.calc_BLEU(pred, rec['true_answer'])

                results[q_type].append(rec)

        return results


    def analyze_results(self, results):
        """결과 분석 및 요약"""
        analysis = {}
        
        for question_type in ['선다형', '단답형', '서술형']:
            if not results[question_type]:
                continue
                
            type_data = results[question_type]
            analysis[question_type] = {}
            
            # for prompt_name in ['baseline', 'simple', 'rich', 'expert', 'format_aware', 'detailed']:
            for prompt_name in ['_'.join(x.split('_')[:-1]) for x in type_data[0].keys() if x.endswith('_pred')]:
                if question_type == "선다형":
                    print(type_data[0].keys())
                    scores = [item[f'{prompt_name}_score'] for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'accuracy': np.mean(scores),
                        'count': len(scores)
                    }
                    
                elif question_type == "단답형":
                    exact_scores = [item[f'{prompt_name}_exact'] for item in type_data]
                    # partial_scores = [item[f'{prompt_name}_partial'] for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'exact_match': np.mean(exact_scores),
                        # 'partial_match': np.mean(partial_scores),
                        'count': len(exact_scores)
                    }
                    
                else:  # 서술형
                    rouge1_scores = [item[f'{prompt_name}_rouge1'] for item in type_data]
                    rouge2_scores = [item[f'{prompt_name}_rouge2'] for item in type_data]
                    rougeL_scores = [item[f'{prompt_name}_rougeL'] for item in type_data]
                    bleu_scores     = [item[f'{prompt_name}_bleu']      for item in type_data]
                    # bert_scores     = [item[f'{prompt_name}_bertscore'] for item in type_data]
                    # bleurt_scores   = [item[f'{prompt_name}_bleurt']   for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'rouge1': np.mean(rouge1_scores),
                        'rouge2': np.mean(rouge2_scores),
                        'rougeL': np.mean(rougeL_scores),
                        'bleu': np.mean(bleu_scores),
                        # 'bertscore': np.mean(bert_scores),
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
                    print(f"{prompt_name:15}: Exact = {metrics['exact_match']:.3f}, (n={metrics['count']})")
                    # print(f"{prompt_name:15}: Exact = {metrics['exact_match']:.3f}, Partial = {metrics['partial_match']:.3f} (n={metrics['count']})")
                    
            else:  # 서술형
                for prompt_name, metrics in type_results.items():
                    print(
                        f"{prompt_name:15}: "
                        f"ROUGE-1 = {metrics['rouge1']:.3f}, "
                        f"ROUGE-2 = {metrics['rouge2']:.3f}, "
                        f"ROUGE-L = {metrics['rougeL']:.3f}, "
                        f"BLEU = {metrics['bleu']:.3f}, "
                        # f"BERTScore = {metrics['bertscore']:.3f}, "
                        # f"BLEURT = {metrics['bleurt']:.3f} "
                        f"(n={metrics['count']})"
                    )
        
        # 전체 평균 (가중 평균)
        print(f"\n🏆 종합 순위:")
        print("-" * 30)
        
        prompt_scores = {}
        for prompt_name in analysis['선다형'].keys():
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
        with open(f'results/phase1_{self.model_name.split("/")[-1]}_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 분석 요약
        with open(f'results/phase1_{self.model_name.split("/")[-1]}_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장 완료:")
        print(f"   - phase1_{self.model_name.split('/')[-1]}_detailed_results.json")
        print(f"   - phase1_{self.model_name.split('/')[-1]}_analysis_summary.json")

def main():
    """메인 실행 함수"""
    print("🚀 Phase 1: Prompting Experiment with Kanana 8B")
    print("="*60)
    
    # 실험 초기화
    experiment = PromptingExperiment("beomi/Kanana-8B")
    
    # 데이터 로드
    train_data, dev_data, test_data = experiment.load_data()
    
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