# Model
maywell/Synatra-7B-Instruct-v0.2

# Modeling flow
- 환경 설정 : cat requirements.txt | xargs -n 1 pip install
- 학습 : sh run.sh
- 추론 : CUDA_VISIBLE_DEVICES=0 python3 inference.py --model_path /results/maywell/Synatra-7B-Instruct-v0.2/trial-0
- 추론 이후 final_submission.ipynb 실행


# Instruction
## train.py
- baseline code 기반

## inference.py
- baseline code 기반
- single prompt tuning 추가

  `prompt_ex2`
    
     "For example, if the choices are ['a, a', 'a,b', 'b,a', 'b,b'] and the correct answer is 'b,a', the correct output should be 2."
    
  `prompt_ex3`
    
    """When the sentence is long, it is divided using "", or ', and each part is labeled from 0 to 3. Provide the correct number that represents the correct part."""

## utils.py
- baseline code 기반

## final_submission.ipynb
- submission의 answer col 후처리
- answer가 0, 1, 2, 3 중 하나가 아닐 경우, 생성된 답변을 토대로 0, 1, 2, 3을 출력할 수 있도록 조정 

