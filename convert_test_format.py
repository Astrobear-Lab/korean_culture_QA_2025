import json
from collections import Counter

# 경로 설정
input_file = "/workspace/korean_culture_QA_2025/data/test.json"         # answer 없는 입력
reference_file = "/workspace/korean_culture_QA_2025/results/test/prompt_v2/phase1_A.X-4.0-Light_test_outputs.json" # answer 포함된 예측 결과
output_file = "/workspace/korean_culture_QA_2025/results/test/prompt_v2/phase1_A.X-4.0-Light_test_outputs_ex3.json"

# 다수결 answer 추출
def get_majority_answer(item):
    answers = [
        item.get("rich_pred"),
        item.get("format_aware_pred"),
        item.get("detailed_pred")
    ]
    count = Counter(answers)
    most_common = count.most_common()
    if most_common[0][1] >= 2:
        return most_common[0][0]
    else:
        return item.get("detailed_pred")

# 전체 처리
def main():
    # 입력 데이터 로드 (정답 없는 쪽)
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 참조 데이터 로드 (정답 후보 포함된 쪽)
    with open(reference_file, "r", encoding="utf-8") as f:
        reference_all = json.load(f)

    # 모든 유형(선다형, 단답형, 서술형) 합치기
    ref_dict = {}
    for qtype in ["선다형", "단답형", "서술형"]:
        for item in reference_all.get(qtype, []):
            ref_dict[item["id"]] = item

    # 정답 추가
    for item in input_data:
        qid = item["id"]
        matched = ref_dict.get(qid)

        if matched:
            item["output"] = {"answer": get_majority_answer(matched)}
        else:
            item["output"] = {"answer": None}  # 또는 "" 등 처리 방식 결정

    # 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)

    print(f"총 {len(input_data)}개 항목 변환 완료 → {output_file}")

if __name__ == "__main__":
    main()
