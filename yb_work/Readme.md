# model
maywell/Synatra-7B-Instruct-v0.2

# config
-- <- 기본 config그냥 가져가요

# Instruction
## train.py
- baseline code 기반

## inference.py
- baseline code 기반
- single prompt tuning 추가
    - prompt_ex2= "For example, if the choices are ['a, a', 'a,b', 'b,a', 'b,b'] and the correct answer is 'b,a', the correct output should be 2."
    - prompt_ex3= """When the sentence is long, it is divided using "", or ', and each part is labeled from 0 to 3. Provide the correct number that represents the correct part."""
- 답안 후처리 추가 

## utils.py
- baseline code 기반

