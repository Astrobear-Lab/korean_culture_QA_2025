import json
import random
from pathlib import Path

# 파일 경로
data_dir = Path("data")
train_path = data_dir / "train.json"
dev_path = data_dir / "dev.json"
test_path = data_dir / "test.json"

# 비율 설정
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1
random_seed = 42

# 데이터 로드 함수
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 데이터 합치기
all_data = load_json(train_path) + load_json(dev_path) + load_json(test_path)

# id 기준 중복 제거
unique = {}
for item in all_data:
    unique[item["id"]] = item
data = list(unique.values())

# 셔플
random.seed(random_seed)
random.shuffle(data)

# 분할
n = len(data)
n_train = int(n * train_ratio)
n_dev = int(n * dev_ratio)

train_data = data[:n_train]
dev_data = data[n_train:n_train + n_dev]
test_data = data[n_train + n_dev:]

# 결과 출력
print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
# 이제 train_data, dev_data, test_data 변수에 각각 분할된 데이터가 담겨 있음 