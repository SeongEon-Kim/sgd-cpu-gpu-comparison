import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_cuda_output(filename):
    """CUDA 출력 파일 파싱"""
    iterations = []
    train_rmse = []
    val_rmse = []
    execution_times = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Iteration' in line and 'RMSE train' in line:
                # Iteration 1/200 RMSE train: 1.142322 -- RMSE val: 1.140894
                match = re.search(r'Iteration (\d+)/\d+ RMSE train: ([\d.]+) -- RMSE val: ([\d.]+)', line)
                if match:
                    iter_num = int(match.group(1))
                    train = float(match.group(2))
                    val = float(match.group(3))
                    iterations.append(iter_num)
                    train_rmse.append(train)
                    val_rmse.append(val)
            elif 'Iteration execution time:' in line:
                # Iteration execution time: 0.179792
                match = re.search(r'Iteration execution time: ([\d.]+)', line)
                if match:
                    time = float(match.group(1))
                    execution_times.append(time)
    
    return iterations, train_rmse, val_rmse, execution_times

def parse_sgd_time_output(filename):
    """SGD 시간 측정 출력 파일 파싱"""
    iterations = []
    train_rmse = []
    val_rmse = []
    execution_times = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Iteration' in line and 'RMSE train' in line:
                # Iteration 1/200 RMSE train: 1.304899 -- RMSE val: 1.301639
                match = re.search(r'Iteration (\d+)/\d+ RMSE train: ([\d.]+) -- RMSE val: ([\d.]+)', line)
                if match:
                    iter_num = int(match.group(1))
                    train = float(match.group(2))
                    val = float(match.group(3))
                    iterations.append(iter_num)
                    train_rmse.append(train)
                    val_rmse.append(val)
            elif 'Iteration execution time:' in line:
                # Iteration execution time: 0.179792
                match = re.search(r'Iteration execution time: ([\d.]+)', line)
                if match:
                    time = float(match.group(1))
                    execution_times.append(time)
    
    return iterations, train_rmse, val_rmse, execution_times

def create_convergence_comparison():
    """Time to Convergence 비교 그래프 생성"""
    
    # 데이터 파싱
    print("CUDA 데이터 파싱 중...")
    cuda_iter, cuda_train, cuda_val, cuda_times = parse_cuda_output('cuda_output_time_128_200.txt')
    
    print("SGD 시간 측정 데이터 파싱 중...")
    sgd_iter, sgd_train, sgd_val, sgd_times = parse_sgd_time_output('sgd_output_time_128_200.txt')
    
    # 이미 누적 시간이므로 cumsum 불필요 (출력 파일에서 이미 누적되어 있음)
    cuda_cumulative_time = np.array(cuda_times)
    sgd_cumulative_time = np.array(sgd_times)
    
    print(f"CUDA: {len(cuda_iter)} iterations, 최종 시간: {cuda_cumulative_time[-1]:.3f}초")
    print(f"SGD: {len(sgd_iter)} iterations, 최종 시간: {sgd_cumulative_time[-1]:.3f}초")
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training MSE
    ax1.plot(cuda_cumulative_time, cuda_train, 'c-', linewidth=1.5, label='CUDA', alpha=0.8)
    ax1.plot(sgd_cumulative_time, sgd_train, 'r-', linewidth=1.5, label='CPU', alpha=0.8)
    ax1.set_xlabel('Execution time (seconds)')
    ax1.set_ylabel('Training MSE')
    ax1.set_title('Training MSE Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation MSE
    ax2.plot(cuda_cumulative_time, cuda_val, 'c-', linewidth=1.5, label='CUDA', alpha=0.8)
    ax2.plot(sgd_cumulative_time, sgd_val, 'r-', linewidth=1.5, label='CPU', alpha=0.8)
    ax2.set_xlabel('Execution time (seconds)')
    ax2.set_ylabel('Validation MSE')
    ax2.set_title('Validation MSE Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 전체 제목
    plt.suptitle('Time to Convergence — Sequential vs CUDA (Batch size = 128)', 
                 fontsize=16, fontweight='bold')
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # plots 폴더에 저장
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/convergence_comparison_time_200.png', dpi=300, bbox_inches='tight')
    
    print("그래프가 plots/convergence_comparison_time.png에 저장되었습니다.")
    
    # 성능 비교 통계 출력
    print("\n=== 성능 비교 통계 ===")
    print(f"CUDA 최종 Training MSE: {cuda_train[-1]:.6f}")
    print(f"CPU 최종 Training MSE: {sgd_train[-1]:.6f}")
    print(f"CUDA 최종 Validation MSE: {cuda_val[-1]:.6f}")
    print(f"CPU 최종 Validation MSE: {sgd_val[-1]:.6f}")
    print(f"CUDA 총 실행 시간: {cuda_cumulative_time[-1]:.3f}초")
    print(f"CPU 총 실행 시간: {sgd_cumulative_time[-1]:.3f}초")
    print(f"속도 향상: {sgd_cumulative_time[-1]/cuda_cumulative_time[-1]:.2f}x")
    
    plt.show()

def create_iteration_comparison():
    """Iteration 기준 비교 그래프도 생성"""
    
    # 데이터 파싱
    cuda_iter, cuda_train, cuda_val, cuda_times = parse_cuda_output('cuda_output_time_128_200.txt')
    sgd_iter, sgd_train, sgd_val, sgd_times = parse_sgd_time_output('sgd_output_time_128_200.txt')
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training MSE (iteration 기준)
    ax1.plot(cuda_iter, cuda_train, 'c-', linewidth=1.5, label='CUDA', alpha=0.8)
    ax1.plot(sgd_iter, sgd_train, 'r-', linewidth=1.5, label='CPU', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training MSE')
    ax1.set_title('Training MSE Convergence (by Iteration)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation MSE (iteration 기준)
    ax2.plot(cuda_iter, cuda_val, 'c-', linewidth=1.5, label='CUDA', alpha=0.8)
    ax2.plot(sgd_iter, sgd_val, 'r-', linewidth=1.5, label='CPU', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Validation MSE')
    ax2.set_title('Validation MSE Convergence (by Iteration)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 전체 제목
    plt.suptitle('Convergence Comparison — Sequential vs CUDA (by Iteration)', 
                 fontsize=16, fontweight='bold')
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # plots 폴더에 저장
    plt.savefig('plots/convergence_comparison_iteration_200.png', dpi=300, bbox_inches='tight')
    
    print("Iteration 기준 그래프가 plots/convergence_comparison_iteration.png에 저장되었습니다.")
    
    plt.show()

if __name__ == "__main__":
    print("=== SGD vs CUDA 성능 비교 그래프 생성 ===")
    
    # 파일 존재 확인
    if not os.path.exists('cuda_output_time_128_200.txt'):
        print("오류: cuda_output_time_128_200.txt 파일을 찾을 수 없습니다.")
        exit(1)
    
    if not os.path.exists('sgd_output_time_128_200.txt'):
        print("오류: sgd_output_time_128_200.txt 파일을 찾을 수 없습니다.")
        exit(1)
    
    # 시간 기준 비교 그래프 생성
    create_convergence_comparison()
    
    # Iteration 기준 비교 그래프 생성
    create_iteration_comparison()
    
    print("\n모든 그래프가 성공적으로 생성되었습니다!")
