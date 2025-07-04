

```
# v1
        
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

3. **서술형 (Descriptive Answer)**  
- 500자 이내로 **신뢰할 수 있고 일관성 있는 문장으로 설명**하십시오."""

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
    format_instruction = "5어절 이내의 **간결하고 정확한 명사 또는 구**로 답하십시오."
else:  # 서술형
    format_instruction = "500자 이내로 **신뢰할 수 있고 일관성 있는 문장으로 설명**하십시오."

prompts['format_aware'] = {"system_prompt":system_prompt, "user_prompt":f"[{question_type}] {question}\n\n{format_instruction}\n답변:"}

prompts['detailed'] = {"system_prompt":detailed_system_prompt, "user_prompt":f"category: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n질문: {question}\n답변:"}
```

```
# v2


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
```