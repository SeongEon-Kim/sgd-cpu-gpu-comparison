#!/usr/bin/env python3
"""
SGD RMSE 시각화 스크립트
SGD 실행 결과에서 RMSE 값을 추출하여 그래프로 시각화합니다.
"""
import matplotlib.pyplot as plt
import re
import sys

def parse_sgd_output(output_file='sgd_output.txt'):
    """SGD 출력에서 RMSE 값을 추출"""
    iterations = []
    rmse_train = []
    rmse_val = []

    with open(output_file, 'r') as f:
        for line in f:
            # "Iteration 1/200 RMSE train: 1440.125251 -- RMSE val: 1436.162098" 형식 파싱
            match = re.search(r'Iteration (\d+)/\d+ RMSE train: ([\d.]+) -- RMSE val: ([\d.]+)', line)
            if match:
                iteration = int(match.group(1))
                train_rmse = float(match.group(2))
                val_rmse = float(match.group(3))

                iterations.append(iteration)
                rmse_train.append(train_rmse)
                rmse_val.append(val_rmse)

    return iterations, rmse_train, rmse_val

def plot_rmse(iterations, rmse_train, rmse_val, output_path='plots/rmse_evolution.png'):
    """RMSE evolution 그래프 생성"""
    plt.figure(figsize=(10, 6))

    # Training RMSE (빨간색)
    plt.plot(iterations, rmse_train, 'r-', linewidth=1.5, label='Train RMSE')

    # Validation RMSE (파란색 점선 - 추가)
    plt.plot(iterations, rmse_val, 'b--', linewidth=1.5, label='Validation RMSE')

    # 축 설정
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)

    # 격자 표시
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 범위 자동 설정 (데이터에 맞춰)
    # 사용자가 0.9-1.3 범위를 요청했지만, 실제 데이터는 1300대이므로 자동 조정
    plt.xlim(0, max(iterations) if iterations else 200)

    # Y축 범위를 데이터에 맞게 자동 조정
    if rmse_train:
        min_rmse = min(min(rmse_train), min(rmse_val)) * 0.95
        max_rmse = max(max(rmse_train), max(rmse_val)) * 1.05
        plt.ylim(min_rmse, max_rmse)

    # 배경색
    plt.gca().set_facecolor('white')

    # 범례 추가
    plt.legend(loc='upper right', fontsize=10)

    # 레이아웃 조정
    plt.tight_layout()

    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # 통계 출력
    print(f"\nRMSE Statistics:")
    print(f"  Training RMSE: {min(rmse_train):.2f} (min) -> {rmse_train[-1]:.2f} (final)")
    print(f"  Validation RMSE: {min(rmse_val):.2f} (min) -> {rmse_val[-1]:.2f} (final)")
    print(f"  Total iterations: {len(iterations)}")

if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'sgd_output.txt'

    try:
        iterations, rmse_train, rmse_val = parse_sgd_output(output_file)

        if not iterations:
            print("Error: No RMSE data found in output file")
            sys.exit(1)

        plot_rmse(iterations, rmse_train, rmse_val)

    except FileNotFoundError:
        print(f"Error: File '{output_file}' not found")
        print("Usage: python plot_rmse.py [output_file]")
        sys.exit(1)
