import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import re
from collections import defaultdict
import torch
from .MUL_main_Infer import inference

def generate_summ_edu_list(Parser, parser_tokenizer, summarizer_tokenizer, Raw_Sentences, batch_size=1):
    try:
        _, EDU_breaks, Parsing_Tree, Relation_Values = inference(Parser, parser_tokenizer, Raw_Sentences, batch_size=batch_size)

        RST_Tokenized = [parser_tokenizer.tokenize(i, add_special_tokens=False) for i in Raw_Sentences]
        SUMM_Tokenized = [summarizer_tokenizer.tokenize(i, add_special_tokens=False) for i in Raw_Sentences]
        SUMM_Tokenized = [token.replace("Ġ", "") for token in SUMM_Tokenized[0]]

        RST_EDU_List = []
        SUMM_EDU_Breaks = []
        SUMM_EDU_List = []

        for index in range(len(EDU_breaks)):
            if index == 0:
                RST_EDU_List.append(RST_Tokenized[0][0:EDU_breaks[index]+1])
            else:
                RST_EDU_List.append(RST_Tokenized[0][EDU_breaks[index-1]+1:EDU_breaks[index]+1])

        for i in range(len(SUMM_Tokenized)):
            if i == 0:
                continue
            elif len(RST_EDU_List) == 0:
                break
            elif SUMM_Tokenized[i] == RST_EDU_List[0][-1].replace("_",""):
                RST_EDU_List.pop(0)
                SUMM_EDU_Breaks.append(i)
            elif SUMM_Tokenized[i] in RST_EDU_List[0][-1].replace("_",""):
                RST_EDU_List.pop(0)
                SUMM_EDU_Breaks.append(i)
            elif RST_EDU_List[0][-1].replace("_","") in SUMM_Tokenized[i]:
                RST_EDU_List.pop(0)
                SUMM_EDU_Breaks.append(i)
            else:
                continue

        for index in range(len(EDU_breaks)):
            if index == 0:
                SUMM_EDU_List.append(SUMM_Tokenized[0:SUMM_EDU_Breaks[index]+1])
            else:
                SUMM_EDU_List.append(SUMM_Tokenized[SUMM_EDU_Breaks[index-1]+1:SUMM_EDU_Breaks[index]+1])
        
        return SUMM_EDU_List, Parsing_Tree, Relation_Values
    except Exception as e:
        # print("An exception occurred:", str(e))
        return SUMM_Tokenized, [], []


def calculate_edu_weights(SUMM_EDU_List, Parsing_Tree, Relation_Values):
    edu_weights = [0] * len(SUMM_EDU_List)
    for i, tree in enumerate(Parsing_Tree):
        nodes_roles = re.findall(r'(\d+):(Nucleus|Satellite)=\w+:(\d+)', tree)
        for node, role, _ in nodes_roles:
            node = int(node) - 1
            if role == 'Nucleus':
                edu_weights[node] = Relation_Values[i]
            else:
                edu_weights[node] = 1 - Relation_Values[i]
    return edu_weights


def generate_token_weight_matrix(Parser, parser_tokenizer, summarizer_tokenizer, Raw_Sentences, dimension):
    try:
        SUMM_EDU_List, Parsing_Tree, Relation_Values = generate_summ_edu_list(Parser, parser_tokenizer, summarizer_tokenizer, Raw_Sentences, batch_size=1)
        edu_weights = calculate_edu_weights(SUMM_EDU_List, Parsing_Tree, Relation_Values)
        total_token_count = sum(len(edu) for edu in SUMM_EDU_List)
        weight_matrix = np.zeros((total_token_count, dimension))
        token_index = 0
        for i, edu in enumerate(SUMM_EDU_List):
            for token in edu:
                weight_matrix[token_index] = [edu_weights[i]] * dimension
                token_index += 1
        weight_matrix = torch.from_numpy(weight_matrix)
    except Exception as e:
        weight_matrix = torch.ones((len(SUMM_Tokenized), dimension))
    return weight_matrix