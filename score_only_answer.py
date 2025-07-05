from __init__ import *


def file_merge(json_path):

    df= pd.read_json('data/test_with_answers.json')

    # JSON 파일 열기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for idx, row in df.iterrows():
        if row['question_type'] == "선다형":
            for item in data['선다형']:
                if item['question'] == row['question']:
                    item['true_answer'] = row['only_answer']
        elif row['question_type'] == "단답형":
            for item in data['단답형']:
                if item['question'] == row['question']:
                    item['true_answer'] = row['only_answer']
        elif row['question_type'] == "서술형":
            for item in data['서술형']:
                if item['question'] == row['question']:
                    item['true_answer'] = row['only_answer']

    return data


def main():
    parser = argparse.ArgumentParser(description="Score-only evaluator for test.json")
    parser.add_argument("--json_path", type=str, help="Path to test.json file with model outputs and answers")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"❌ 파일이 존재하지 않습니다: {args.json_path}")
        return

    experiment = PromptingExperiment(load_model=False)
    results = experiment.only_scoring(file_merge(args.json_path))
    print(results)
    print('--')
    analysis = experiment.analyze_results(results)
    print(analysis)
    experiment.print_analysis(analysis)


if __name__ == "__main__":
    main()
