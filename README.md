# Phase 1: Prompting Experiment (Kanana 1.5 Instruct)

Kanana 1.5 8B Instruct 모델(`kakaocorp/kanana-1.5-8b-instruct-2505`)로 5가지 프롬프트 전략을 테스트하는 실험입니다.

## 🎯 실험 목표

한국 문화 QA 데이터셋에서 어떤 프롬프트 전략이 가장 효과적인지 평가합니다.

## 📋 테스트하는 프롬프트

1. **Baseline**: 질문만 제공
   ```
   system : "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요."
   user : "질문: {question}\n답변:"
   ```

2. **Simple**: 질문 유형 + 질문
   ```
   system : "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요."
   user :"[{question_type}] {question}\n답변:"
   ```

3. **Rich**: 모든 메타데이터 포함
   ```
   system : "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요."
   user : "분류: {category}\n도메인: {domain}\n주제: {topic_keyword}\n답변 유형: {question_type}\n\n질문: {question}\n답변:"
   ```

4. **Expert**: 전문가 역할 부여
   ```
   system : "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요."
   user : "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요.\n질문: {question}\n답변:"
   ```

5. **Format-aware**: 답변 형식 명시
   ```
   system : "당신은 한국 문화 전문가입니다. 다음 질문에 정확하고 적절하게 답변해주세요."
   user :"[{question_type}] {question}\n\n{format_instruction}\n답변:"
   ```
6. **detailed** : 자세한 설명 추가
   ```
   system : "당신은 다양한 문화 지식, 관점, 실행에 기반하여 질문에 신뢰도 높고 정확한 답변을 생성하는 한국어 전문가 AI입니다.

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
   
   3. **서술형 (Descriptive Answer)**  
      - 500자 이내로 **신뢰할 수 있고 일관성 있는 문장으로 설명**하십시오."
   user : "category: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n질문: {question}\n답변:"
   ```

## 📊 평가 지표

- **선다형**: Accuracy
- **단답형**: Exact Match, Partial Match
- **서술형**: ROUGE-1, ROUGE-2, ROUGE-L

## 🚀 실행 방법

### 1. 패키지 설치
```bash
pip install -r requirements_phase1.txt
```

### 2. 모델명 확인
- `phase1_prompting_experiment.py`에서 모델명을 아래와 같이 사용:
  ```python
  model_name = "kakaocorp/kanana-1.5-8b-instruct-2505"
  ```

### 3. 기본 실행 (dev set 사용)
```bash
python run_phase1.py
```

### 4. 다양한 옵션으로 실행
```bash
python run_phase1.py --use_train
python run_phase1.py --sample_size 20
python run_phase1.py --model "kakaocorp/kanana-1.5-8b-instruct-2505"
```

### 5. 빠른 테스트
```bash
python quick_test_phase1.py
```

## 📁 출력 파일

실험 완료 후 다음 파일들이 생성됩니다:

- `phase1_detailed_results.json`: 모든 샘플의 상세 결과
- `phase1_analysis_summary.json`: 프롬프트별 성능 요약
- `phase1_intermediate_results_*.json`: 중간 저장 파일들 (10개씩 처리할 때마다)

## 🔧 커스터마이징

### 프롬프트 수정
`phase1_prompting_experiment.py`의 `create_prompts()` 메서드를 수정하여 새로운 프롬프트를 추가하거나 기존 프롬프트를 변경할 수 있습니다.

### 평가 지표 추가
각 질문 유형별 평가 메서드(`evaluate_multiple_choice`, `evaluate_short_answer`, `evaluate_long_answer`)를 수정하여 추가 지표를 계산할 수 있습니다.

### 모델 변경
다른 한국어 LLM으로 실험하려면 `--model` 옵션을 사용하거나 `PromptingExperiment` 클래스 초기화 시 모델명을 변경하세요.

## ⚠️ Apple Silicon(M1/M2/M3) 환경 주의사항
- 8B 모델은 Apple Silicon(M3 Max 포함)에서 GPU(mps)로 실행이 거의 불가능합니다.
- 실험은 CPU로만 가능할 확률이 높으며, 실행 속도가 매우 느릴 수 있습니다.
- Colab, Lambda, Paperspace 등 NVIDIA GPU 환경에서 실험하는 것을 권장합니다.

## 🛠️ 코드 주요 변경점
- 모델명: `kakaocorp/kanana-1.5-8b-instruct-2505` (Q&A에 최적화된 instruct 버전)
- KeyError('단답형') 오류 수정: `results` 딕셔너리를 질문 유형별로 초기화
- bfloat16 지원 (단, Apple Silicon에서는 float16/float32만 지원)

## 📈 결과 해석
