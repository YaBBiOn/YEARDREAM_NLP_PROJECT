import pandas as pd

import re
import ast

# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer

import pickle
import json
import os

def col_and_clustering(df):
    # 새로운 컬럼 생성
    df['cluster'] = 0
    df['__선택지'] = '0'
    df['0'] = 0
    df['1'] = 0
    df['2'] = 0
    df['3'] = 0

    # 문제 클러스터링
    ## '문제' 컬럼이 '?'로 끝나는 행들의 'cluster' 값을 1로 변경
    df.loc[df['문제'].str.endswith('?'), 'cluster'] = 1
    ## '문제' 컬럼이 '시나리오 1 |'을 포함하는 행들의 'cluster' 값을 2로 변경
    df.loc[df['문제'].str.contains('시나리오 1 |', regex=False), 'cluster'] = 2
    ## '문제' 컬럼이 '___'을 포함하는 행들의 'cluster' 값을 3으로 변경
    df.loc[df['문제'].str.contains('___', regex=False), 'cluster'] = 3
    ## 명제(진술)이 참인지 거짓인지 판단하는 문제
    df.loc[df['문제'].str.contains('명제 1 |',regex=False), 'cluster'] = 4
    ## 정보를 읽고 그에 맞게 답하는 문제
    df.loc[df['문제'].str.contains('다음 정보', regex=False), 'cluster'] = 5

    # 3번문제 한정 데이터 전처리 : ___ -> <BLANK>
    subset = df.loc[df['cluster'] == 3, '문제']
    df.loc[df['cluster'] == 3, '문제'] = subset.str.replace(r'_{2,}', '<BLANK>', regex=True)

    return df

def allocate(df):
    # 선택지 리스트화 + 리스트화된 선택지를 항목별로 쪼갬 + 각 번호에 맞는 컬럼에 넣음

    for index, row in df.iterrows():
        try:
            df['__선택지'][index] = ast.literal_eval(df['선택지'][index])
        except Exception as e:
            # print(f'{index} error 발생 : {e}')
            # print("원본 선택지:", df['선택지'][index])
            # 에러 발생 시 원본 선택지를 그대로 사용
            df['__선택지'][index] = df['선택지'][index]

        try:
            df['0'][index] = df['__선택지'][index][0]
        except:
            df['0'][index] = '0'

        try:
            df['1'][index] = df['__선택지'][index][1]
        except:
            df['1'][index] = '0'

        try:
            df['2'][index] = df['__선택지'][index][2]
        except:
            df['2'][index] = '0'

        try:
            df['3'][index] = df['__선택지'][index][3]
        except:
            df['3'][index] = '0'
    
    # 선택지 쪼개진걸 기반으로 문항개수 컬럼 추가
    df['문항개수'] = df['__선택지'].apply(lambda x: len(x))

    return df

def save_train(df):
    df.to_csv(os.path.join(BASE_DIR, 'data/temp_train.csv'), encoding='utf-8')
    if os.path.exists(os.path.join(BASE_DIR, 'data/train_data.csv')):
        os.remove(os.path.join(BASE_DIR, 'data/train_data.csv'))
    os.rename(os.path.join(BASE_DIR, 'data/temp_train.csv'), os.path.join(BASE_DIR, 'data/train_data.csv'))

    print('preprocessing for train dataset is complete')

def save_test(df):
    df.to_csv(os.path.join(BASE_DIR, 'data/temp_test.csv'), encoding='utf-8')
    if os.path.exists(os.path.join(BASE_DIR, 'data/ttest_data.csv')):
        os.remove(os.path.join(BASE_DIR, 'data/test_data.csv'))
    os.rename(os.path.join(BASE_DIR, 'data/temp_test.csv'), os.path.join(BASE_DIR, 'data/test_data.csv'))

    print('Preprocessing for test dataset is complete')

if __name__ == "__main__":
    # set base directory 
    BASE_DIR = os.path.dirname(__file__)

    train = pd.read_csv(os.path.join(BASE_DIR, 'data/train_data.csv'), encoding='utf-8')
    print('Loading of original train dataset is complete')
    
    print('Preprocessing for train dataset...')
    train = col_and_clustering(train)
    train = allocate(train)
    save_train(train)
    

    test = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    print('Loading of original test dataset is complete')

    print('Preprocessing for test dataset...')
    test = col_and_clustering(test)
    test = allocate(test)
    save_test(test)
    
    print('Preprocessing is complete!')
    
