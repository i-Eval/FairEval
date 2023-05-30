m1=gpt35
m2=vicuna-13b
eval_model=gpt-3.5-turbo-0301
bpc=1
k=3
python3 FairEval.py \
    -q question.jsonl \
    -a answer/answer_$m1.jsonl answer/answer_$m2.jsonl \
    -o review/review_${m1}_${m2}_${eval_model}_mec${k}_bpc${bpc}.json \
    -m $eval_model \
    --bpc $bpc \
    -k $k