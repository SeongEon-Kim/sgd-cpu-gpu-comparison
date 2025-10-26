#! /bin/bash
# Python으로 전처리 수행 (R 대체)
python3 preprocess_flights.py

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
