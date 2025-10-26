import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_output(filename):
    iterations = []
    train_rmse = []
    val_rmse = []
    execution_times = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Iteration' in line and 'RMSE train' in line:
                match = re.search(r'Iteration (\d+)/\d+ RMSE train: ([\d.]+) -- RMSE val: ([\d.]+)', line)
                if match:
                    iterations.append(int(match.group(1)))
                    train_rmse.append(float(match.group(2)))
                    val_rmse.append(float(match.group(3)))
            elif 'Iteration execution time:' in line:
                match = re.search(r'Iteration execution time: ([\d.]+)', line)
                if match:
                    execution_times.append(float(match.group(1)))
    
    return iterations, train_rmse, val_rmse, execution_times

# Parse data
cpu_iterations, cpu_train_rmse, cpu_val_rmse, cpu_times = parse_output('sgd_output_time_128_200.txt')
cuda_iterations, cuda_train_rmse, cuda_val_rmse, cuda_times = parse_output('cuda_output_time_128_200.txt')

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Training MSE plot
ax1.plot(cpu_times, cpu_train_rmse, 'r-', linewidth=2, label='CPU', marker='o', markersize=3)
ax1.plot(cuda_times, cuda_train_rmse, 'c-', linewidth=2, label='CUDA', marker='s', markersize=3)
ax1.set_xlabel('Execution time (seconds)')
ax1.set_ylabel('Training MSE')
ax1.set_title('Training MSE - First 12.5s')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 12.5)

# Test MSE plot
ax2.plot(cpu_times, cpu_val_rmse, 'r-', linewidth=2, label='CPU', marker='o', markersize=3)
ax2.plot(cuda_times, cuda_val_rmse, 'c-', linewidth=2, label='CUDA', marker='s', markersize=3)
ax2.set_xlabel('Execution time (seconds)')
ax2.set_ylabel('Test MSE')
ax2.set_title('Test MSE - First 12.5s')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 12.5)

plt.tight_layout()
plt.savefig('plots/early_convergence_12_5s.png', dpi=300, bbox_inches='tight')
plt.show()