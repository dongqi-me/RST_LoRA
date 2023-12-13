import os
import torch
import numpy as np
import argparse
import os
from . import config
from transformers import AutoTokenizer, AutoModel
from . model_depth import ParsingNet

# os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default='depth_mode/Savings/multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []
    all_relation_values_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks,Relation_Values = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
            all_relation_values_pred.extend(Relation_Values)
    return input_sentences, all_segmentation_pred[0], all_tree_parsing_pred[0], all_relation_values_pred



def parser_input(Input_sentences):
    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size

    parser_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    parser_backbone = AutoModel.from_pretrained("xlm-roberta-base")
    parser_backbone = parser_backbone.cuda()
    for name, param in parser_backbone.named_parameters():
        param.requires_grad = False
    model = ParsingNet(parser_backbone, bert_tokenizer=parser_tokenizer)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    Test_InputSentences = Input_sentences
    input_sentences, all_segmentation_pred, all_tree_parsing_pred, all_relation_values_pred = inference(model, parser_tokenizer, Test_InputSentences, batch_size)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred, all_relation_values_pred