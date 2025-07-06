from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# LoRA adapter를 로드 (HuggingFace 혹은 로컬에 위치)
peft_model = AutoPeftModelForCausalLM.from_pretrained("/workspace/korean_culture_QA_2025/models/grpo_v1_A.X-4.0-Light/checkpoint-95")

# LoRA를 base 모델에 병합
merged_model = peft_model.merge_and_unload()

# tokenizer도 base 모델에서 불러와 저장
tokenizer = AutoTokenizer.from_pretrained("/workspace/korean_culture_QA_2025/models/grpo_v1_A.X-4.0-Light/checkpoint-95")
merged_model.save_pretrained("/workspace/korean_culture_QA_2025/models/grpo_v1_A.X-4.0-Light/checkpoint-95_merged")
tokenizer.save_pretrained("/workspace/korean_culture_QA_2025/models/grpo_v1_A.X-4.0-Light/checkpoint-95_merged")
