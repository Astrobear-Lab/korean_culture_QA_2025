# !pip install unsloth unsloth_zoo
from unsloth import FastLanguageModel
import torch
import argparse
import json
import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

def main():
    parser = argparse.ArgumentParser(description="Phase 2: SFT Experiment")
    parser.add_argument(
        "--model", 
        default="kakaocorp/kanana-1.5-8b-instruct-2505", 
        help="Model name to use"
    )
    args = parser.parse_args()

    # 1) 모델 & 토크나이저 로드 (LoRA 포함)
    max_seq_length = 2048
    lora_rank = 32

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = max_seq_length,
        load_in_4bit = False,
        fast_inference = True,
        max_lora_rank = lora_rank,
        # gpu_memory_utilization = 0.8,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2) 시스템 프롬프트
    system_prompt = """당신은 한국의 문화에 기반하여 질문에 신뢰도 높고 정확한 답변을 생성하는 한국어 전문가 AI입니다.

사용자가 입력한 다음 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오:
- 카테고리(category) 및 도메인(domain): 질문이 속한 전반적인 지식 분야
- 주제(topic_keyword): 문제의 핵심 키워드
- 질문 유형(question_type): '선다형', '단답형', 또는 '서술형' 중 하나
- 질문 내용(question): 사용자가 직접 묻는 질문

당신의 답변은 다음과 같은 형식을 따라야 합니다:
1. **선다형 (Multiple Choice)**  
   - 보기 중 정답에 해당하는 번호만 **숫자**로 출력하십시오.

2. **단답형 (Short Answer)**  
   - 5어절 이내의 **명사 또는 구**로 답하십시오.  

3. **서술형 (Descriptive Answer)**  
   - 적절한 답변을 문장으로 설명하십시오."""

    # 3) 데이터 불러오기 및 전처리
    df = pd.read_csv('data/preprocessed/sft_train.csv')
    df['answer']   = df['answer'].astype(str).str.strip()
    df['question'] = df['question'].astype(str).str.strip()
    df['prompt']   = df.apply(
        lambda row: (
            "주어진 질문에 적절한 답변을 해주세요.\n\n"
            f"category: {row['category']}\n"
            f"domain: {row['domain']}\n"
            f"topic_keyword: {row['topic_keyword']}\n"
            f"question_type: {row['question_type']}\n\n"
            f"<질문>\n{row['question']}\n\n"
            f"<답변>"
        ),
        axis=1
    )

    dataset = Dataset.from_pandas(df[["prompt", "answer"]])

    # 4) Chat-format 으로 매핑
    def to_chat_format(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": example["prompt"]}
            ],
            "answer": example["answer"]
        }
    dataset = dataset.map(to_chat_format)

    # 5) 토크나이즈
    def tokenize_fn(x):
        tokens = tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )
        return {"tokens": tokens, "labels": tokenizer.tokenize(x["answer"], add_special_tokens=False)}

    tokenized = dataset.map(tokenize_fn, batched=True)

    # 6) 길이 필터링 (optional)
    import numpy as np
    lengths = [len(x) for x in tokenized["tokens"]]
    max_length = int(np.quantile(lengths, 1.00))
    keep_idxs = [i for i, l in enumerate(lengths) if l <= max_length]
    dataset = dataset.select(keep_idxs)

    # 7) SFTConfig 설정
    num_train_epochs = 3
    training_args = SFTConfig(
        learning_rate = 1e-4,
        weight_decay = 0.01,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        lr_scheduler_type = "linear",
        warmup_ratio = 0.05,
        # logging_steps = 10,
        # save_steps = 100,
        num_train_epochs = num_train_epochs, # Set to 1 for a full training run
        save_steps = 0.49 / num_train_epochs,
        output_dir = "models/sft_"+args.model.split("/")[-1],
        report_to = "wandb",
    )

    # 8) WandB 초기화 (선택)
    import wandb
    wandb.login()  # 환경변수에 API 키 세팅 필요
    wandb.init(
        project="moducorpus_korea_culture",
        name=f"sft_{args.model.split('/')[-1]}"
    )

    # 9) SFTTrainer로 학습
    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    main()
