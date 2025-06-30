import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
from matplotlib import font_manager
import os

# 0️⃣ figures 디렉토리 생성
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# 1️⃣ 한글 폰트 설정 (Glyph Warning 제거 및 한글 깨짐 방지)
font_path = "/Library/Fonts/NanumGothic.ttf"  # 본인 설치 경로 확인 후 사용
font_manager.fontManager.addfont(font_path)
font_prop = font_manager.FontProperties(fname=font_path)
font_name = font_prop.get_name()

plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# 2️⃣ 데이터 로드
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data_dir = Path("data")
train_data = load_json(data_dir / "train.json")
dev_data = load_json(data_dir / "dev.json")
test_data = load_json(data_dir / "test.json")

# 3️⃣ DataFrame 생성
def create_dataframe(data, include_answer=True):
    df_list = []
    for item in data:
        row = {
            'id': item['id'],
            'category': item['input']['category'],
            'domain': item['input']['domain'],
            'question_type': item['input']['question_type'],
            'topic_keyword': item['input']['topic_keyword'],
            'question': item['input']['question'],
            'question_length': len(item['input']['question'])
        }
        if include_answer and 'output' in item:
            row['answer'] = item['output']['answer']
            row['answer_length'] = len(item['output']['answer'])
        df_list.append(row)
    return pd.DataFrame(df_list)

train_df = create_dataframe(train_data, include_answer=True)
dev_df = create_dataframe(dev_data, include_answer=True)
test_df = create_dataframe(test_data, include_answer=False)

print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

# 4️⃣ 그래프 저장 함수
def save_plot(fig, filename, dpi=300):
    fig.tight_layout()
    save_path = figures_dir / f"{filename}.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# 5️⃣ 시각화

# 1) Train set 질문 유형 분포 (파이차트)
fig, ax = plt.subplots(figsize=(10, 8))
question_type_counts = train_df['question_type'].value_counts()
colors = ['#FF9999', '#66B2FF', '#99FF99']
wedges, texts, autotexts = ax.pie(question_type_counts.values,
                                   labels=question_type_counts.index,
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   startangle=90)
ax.set_title('Train Set - Question Type Distribution')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
save_plot(fig, "train_question_type_pie")

# 2) 전체 데이터셋 분할별 질문 유형 분포 (스택 바차트)
fig, ax = plt.subplots(figsize=(12, 8))
split_data = []
for name, df in [('Train', train_df), ('Dev', dev_df), ('Test', test_df)]:
    for qtype in df['question_type'].value_counts().index:
        count = df[df['question_type'] == qtype].shape[0]
        split_data.append({'Split': name, 'Question_Type': qtype, 'Count': count})

split_df = pd.DataFrame(split_data)
pivot_df = split_df.pivot(index='Split', columns='Question_Type', values='Count').fillna(0)
pivot_df.plot(kind='bar', stacked=True, ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
ax.set_title('Question Type Distribution by Dataset Split')
ax.set_xlabel('Dataset Split')
ax.set_ylabel('Number of Questions')
ax.legend(title='Question Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
save_plot(fig, "split_question_type_stacked")

# 3) Train set 도메인 분포 (수평 바차트)
fig, ax = plt.subplots(figsize=(12, 8))
domain_counts = train_df['domain'].value_counts()
bars = ax.barh(range(len(domain_counts)), domain_counts.values, color='skyblue')
ax.set_yticks(range(len(domain_counts)))
ax.set_yticklabels(domain_counts.index)
ax.set_xlabel('Number of Questions')
ax.set_title('Train Set - Domain Distribution')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            f'{int(width)}', ha='left', va='center')
save_plot(fig, "train_domain_distribution")

# 4) Train set 카테고리 분포 (도넛 차트)
fig, ax = plt.subplots(figsize=(10, 8))
category_counts = train_df['category'].value_counts()
colors = ['#FFB366', '#66FFB2', '#B366FF']
wedges, texts, autotexts = ax.pie(category_counts.values,
                                   labels=category_counts.index,
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   pctdistance=0.85)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax.add_artist(centre_circle)
ax.set_title('Train Set - Category Distribution')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
save_plot(fig, "train_category_donut")

# 5) 질문 길이 분포 (히스토그램 및 박스플롯)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# (a) Train 질문 길이 히스토그램
axes[0, 0].hist(train_df['question_length'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
axes[0, 0].axvline(train_df['question_length'].mean(), color='red', linestyle='--',
                   label=f'Mean: {train_df["question_length"].mean():.1f}')
axes[0, 0].set_title('Train Set - Question Length Distribution')
axes[0, 0].set_xlabel('Question Length (characters)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# (b) Split별 질문 길이 박스플롯
all_lengths = [train_df['question_length'], dev_df['question_length'], test_df['question_length']]
labels = ['Train', 'Dev', 'Test']
axes[0, 1].boxplot(all_lengths, labels=labels)
axes[0, 1].set_title('Question Length Comparison by Split')
axes[0, 1].set_ylabel('Question Length (characters)')

# (c) Train 답변 길이 히스토그램 (질문 유형별)
for qtype in ['선다형', '단답형', '서술형']:
    if qtype in train_df['question_type'].values:
        subset = train_df[train_df['question_type'] == qtype]['answer_length']
        if len(subset) > 0:
            axes[1, 0].hist(subset, bins=20, alpha=0.6, label=qtype, edgecolor='black')
axes[1, 0].set_title('Train Set - Answer Length by Question Type')
axes[1, 0].set_xlabel('Answer Length (characters)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')

# (d) 질문 유형별 답변 길이 박스플롯
train_df.boxplot(column='answer_length', by='question_type', ax=axes[1, 1])
axes[1, 1].set_title('Answer Length by Question Type (Train Set)')
axes[1, 1].set_xlabel('Question Type')
axes[1, 1].set_ylabel('Answer Length (characters)')
plt.suptitle('')
save_plot(fig, "length_distributions")

# 6) 도메인별 질문 유형 히트맵
fig, ax = plt.subplots(figsize=(12, 8))
domain_qtype = pd.crosstab(train_df['domain'], train_df['question_type'])
sns.heatmap(domain_qtype, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Train Set - Domain vs Question Type Heatmap')
ax.set_xlabel('Question Type')
ax.set_ylabel('Domain')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
save_plot(fig, "domain_questiontype_heatmap")

# 7) 선다형 문제 답변 분포
fig, ax = plt.subplots(figsize=(10, 6))
mc_answers = train_df[train_df['question_type'] == '선다형']['answer'].value_counts().sort_index()
bars = ax.bar(mc_answers.index, mc_answers.values, color='lightcoral', edgecolor='black')
ax.set_title('Train Set - Multiple Choice Answer Distribution')
ax.set_xlabel('Answer Choice')
ax.set_ylabel('Frequency')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom')
expected = len(train_df[train_df['question_type'] == '선다형']) / len(mc_answers)
ax.axhline(y=expected, color='red', linestyle='--', alpha=0.7,
           label=f'Expected (uniform): {expected:.1f}')
ax.legend()
save_plot(fig, "train_mc_answer_distribution")

# 8) 상위 토픽 키워드
fig, ax = plt.subplots(figsize=(14, 8))
top_topics = train_df['topic_keyword'].value_counts().head(15)
bars = ax.bar(range(len(top_topics)), top_topics.values, color='lightgreen', edgecolor='black')
ax.set_xticks(range(len(top_topics)))
ax.set_xticklabels(top_topics.index, rotation=45, ha='right')
ax.set_title('Train Set - Top 15 Topic Keywords')
ax.set_xlabel('Topic Keywords')
ax.set_ylabel('Frequency')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
            f'{int(height)}', ha='center', va='bottom')
save_plot(fig, "train_top_topics")

# 9) 데이터셋 크기 비교
fig, ax = plt.subplots(figsize=(10, 6))
sizes = [len(train_df), len(dev_df), len(test_df)]
labels = ['Train', 'Dev', 'Test']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(labels, sizes, color=colors, edgecolor='black')
ax.set_title('Dataset Split Sizes')
ax.set_ylabel('Number of Samples')
total = sum(sizes)
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
            f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')
save_plot(fig, "dataset_split_sizes")

# 10) 통계 요약 테이블
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
stats_data = []
for name, df in [('Train', train_df), ('Dev', dev_df), ('Test', test_df)]:
    row = [
        name,
        len(df),
        f"{df['question_length'].mean():.1f}±{df['question_length'].std():.1f}",
        df['category'].nunique(),
        df['domain'].nunique(),
        df['question_type'].nunique(),
        f"{df['answer_length'].mean():.1f}±{df['answer_length'].std():.1f}" if 'answer_length' in df.columns else 'N/A'
    ]
    stats_data.append(row)
columns = ['Split', 'Size', 'Question Length', 'Categories', 'Domains', 'Question Types', 'Answer Length']
table = ax.table(cellText=stats_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax.set_title('Dataset Statistics Summary')
save_plot(fig, "statistics_summary")

# 완료 메시지
print("\n=== 시각화 완료 ===")
for file in figures_dir.glob("*.png"):
    print(f" - {file}")
