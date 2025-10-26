#!/usr/bin/env python3
"""
Kaggle에서 Flight Delays 데이터셋 다운로드
"""
import kagglehub
import shutil
import os

# 데이터셋 다운로드
print("Downloading flight delays dataset from Kaggle...")
path = kagglehub.dataset_download("usdot/flight-delays")
print(f"Dataset downloaded to: {path}")

# data 폴더로 복사
target_dir = "/workspace/SGD/data"
print(f"\nCopying files to {target_dir}...")

# 다운로드된 파일들을 data 폴더로 복사
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(target_dir, file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {file}")

print("\nDownload complete!")
print(f"Files in {target_dir}:")
for file in os.listdir(target_dir):
    file_path = os.path.join(target_dir, file)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  - {file} ({size_mb:.2f} MB)")
