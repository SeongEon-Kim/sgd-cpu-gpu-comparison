#! /bin/bash
# 스크립트 디렉토리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Python으로 전처리 수행 (R 대체)
python3 "$SCRIPT_DIR/preprocess_flights.py"

# 프로젝트 루트로 이동하여 작업
cd "$PROJECT_ROOT"

# Bias term 추가 (각 행 앞에 "1" 추가)
awk -F" " '{print "1",$0}' X_train.txt > X_ent.txt
rm X_train.txt

# 초기 가중치 생성 (39개 = 38 features + 1 bias)
for i in {1..39}
do
   echo 0.1
done > b_bh.txt

# Validation 데이터에도 bias term 추가
awk -F" " '{print "1",$0}' X_val.txt > X_valida.txt
rm X_val.txt

echo "Preprocessing complete! Files created in $PROJECT_ROOT"
