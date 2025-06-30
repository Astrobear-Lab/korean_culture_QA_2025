#!/usr/bin/env python3
"""
Phase 1 빠른 테스트 스크립트
소수의 샘플로 실험을 테스트해볼 수 있습니다.
"""

from phase1_prompting_experiment import PromptingExperiment
import json

def quick_test():
    """5개 샘플로 빠른 테스트"""
    print("🧪 Phase 1 빠른 테스트 시작")
    print("   - 샘플 수: 5개")
    print("   - 목적: 코드 동작 확인")
    print("="*50)
    
    # 실험 초기화
    experiment = PromptingExperiment("beomi/Kanana-8B")
    
    # 데이터 로드
    train_data, dev_data = experiment.load_data()
    
    # 5개 샘플만 사용
    test_data = dev_data[:5]
    
    print(f"\n🔬 테스트 시작 (n={len(test_data)})")
    print("샘플 정보:")
    for i, sample in enumerate(test_data):
        print(f"  {i+1}. [{sample['input']['question_type']}] {sample['input']['question'][:50]}...")
    
    # 실험 실행
    results = experiment.run_experiment(test_data, save_results=False)
    
    # 결과 분석
    analysis = experiment.analyze_results(results)
    
    # 결과 출력
    experiment.print_analysis(analysis)
    
    # 샘플 결과 출력
    print(f"\n📝 샘플 결과 예시:")
    print("-" * 50)
    
    for question_type in ['선다형', '단답형', '서술형']:
        if results[question_type]:
            sample = results[question_type][0]  # 첫 번째 샘플
            print(f"\n[{question_type}] 질문: {sample['question'][:100]}...")
            print(f"정답: {sample['true_answer']}")
            print("예측:")
            for prompt_name in ['baseline', 'expert', 'format_aware']:
                pred = sample.get(f'{prompt_name}_pred', 'N/A')
                print(f"  {prompt_name:12}: {pred[:100]}...")
            break
    
    print(f"\n✅ 빠른 테스트 완료!")
    print(f"💡 전체 실험을 실행하려면: python run_phase1.py")

if __name__ == "__main__":
    quick_test() 