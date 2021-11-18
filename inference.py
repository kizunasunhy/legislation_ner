import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from model import Bert_CRF, Roberta_CRF
from utils import define_label

import torch
import numpy as np
import os
import argparse

def inference_single(tokenizer, model, input_list, crf=False):
    print(' '.join(input_list))
    test_encodings = tokenizer(input_list, is_split_into_words=True,
                               return_offsets_mapping=True, truncation=True)
    model.eval()

    output = model(torch.tensor([test_encodings["input_ids"]]).cuda(),
               attention_mask=torch.tensor([test_encodings['attention_mask']]).cuda())
    res = output[0].argmax(dim=2) #for bert output
    if crf:
        predictions = model.crf.decode(output[0]) 
        print('CRF predicts: ', predictions[0])
    print('BERT predicts: ', res[0])
    print('-------------------------------------------')
    bert_word_dict = {}
    crf_word_dict = {}
    for n in range(len(res[0])):
        if test_encodings.word_ids()[n] in bert_word_dict or test_encodings.word_ids()[n] is None:
            continue
        bert_word_dict[test_encodings.word_ids()[n]] = int(res[0][n])
    if crf:
        for n in range(len(predictions[0])):
            if test_encodings.word_ids()[n] in crf_word_dict or test_encodings.word_ids()[n] is None:
                continue
            crf_word_dict[test_encodings.word_ids()[n]] = int(predictions[0][n])  
        #----------------CRF results-----------------
        for i, enc in enumerate(input_list):
            print(i, enc, "\t", label_dict_rev.get(crf_word_dict.get(i, 'NaN'), 'NaN'))
        print('-------------------------------------------')

    #----------------BERT results-----------------
    for i, enc in enumerate(input_list):
        print(i, enc, "\t", label_dict_rev.get(bert_word_dict.get(i, 'NaN'), 'NaN'))
    print('============================================')
    return

def main(parser):
    args = parser.parse_args()
    
    label_list = ['O', 'B-substitute', 'I-substitute', 'B-before-insertions', 'I-before-insertions', 'B-after-insertions', 
              'I-after-insertions', 'B-revocation', 'I-revocation']
    global label_dict_rev
    label_dict, label_dict_rev = define_label(label_list)
    
    model_checkpoint = args.model_checkpoint
    #"bert-base-cased", 'roberta-base', 'microsoft/mpnet-base', 'allenai/longformer-base-4096', 'nghuyong/ernie-2.0-en'
    #'google/bigbird-roberta-base', 'gpt2', 'microsoft/deberta-base', 'roberta-large'
    model_directory = args.model_dir
    model_name = model_checkpoint.split('-')[0].split('/')[-1]

    #-------------------------------LOAD TOKENIZER-------------------------
    if model_name == 'bert' or model_name == 'ernie' or model_name=='mpnet':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if model_name == 'roberta' or model_name=='longformer' or model_name=='bigbird' or model_name=='gpt2' or model_name=='deberta':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True) #For Roberta and Longformer
    if model_name == 'gpt2':
        ADD_SPECIAL_TOKENS = True
        num_added_tokens = tokenizer.add_special_tokens({'pad_token':'<PAD>'})
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    

    #-------------------------------LOAD MODEL-----------------------------
    crf = args.crf
    if crf == True:
        if model_name == bert:
            model = Bert_CRF.from_pretrained(model_directory, num_labels=len(label_list))
        if model_name == roberta:
            model = Roberta_CRF.from_pretrained(model_directory, num_labels=len(label_list))  
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_directory, num_labels=len(label_list))
        if model_name == 'gpt2':
            model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + num_added_tokens)
    device = torch.device('cuda')
    model.to(device)
    model.eval()      
    
    src = 'for the words “Supreme Court Act 1981”, substitute “ , that the decision”'
    src = 'In subsection (2) , omit the words from “that the decision” to the end and insert'
    src = 'in sub-paragraph (6)(a) for the words after “under paragraph (1) or regulation 7(1)” insert “under paragraph (1), regulation 7(1) or regulation 8(1)”;'
    src = 'for the words from “by whom or” to the end, substitute “ to —'
    src = 'In regulation 23A (supply of and charge for test set), in paragraph (1) after “(db),” insert “(dc), (dd),”.'
    src = 'in paragraph (2), for “23(3) or 24(3)” substitute “23(1)(de)(ii), (1)(df)(ii), (3), 24(1)(ce)(ii), (1)(cf)(ii) or (3)”.'
    tar_before = 'after paragraph (e) of Supreme Court Act 1981 insert—“(ea) proceedings under section 79 of the Childcare Act 2006;”'
    tar_before = 'That procedure must be designed to secure , among other things , that the decision which gives rise to the obligation to give any such notice is taken by a person not directly involved in establishing the evidence on which that decision is based.'
    tar_before = 'a prescribed sum is payable by that licensee in respect of a licence under paragraph (1) or regulation 7(1);'
    tar_before ='The Secretary of State may make it a condition of a direction under subsection (1) that any person by whom or with whose agreement the request for the direction was made should , when so directed or at specified intervals , report to the Secretary of State on any matters specified by him.'
    tar_before = 'The Secretary of State shall, on request, supply (by electronic or other means) a test set to any person who has appointed another person or class of persons to conduct theory tests under sub-paragraph (b), (c), (da), (db), (e) or (f) of regulation 23(1) or under regulation 23(2)(b).'
    tar_before = 'Where a person has his appointment revoked or if an approval given in respect of him under regulation 23(3) or 24(3) is withdrawn, that person shall immediately return to the Secretary of State all forms of pass certificates supplied to him under regulations 47(8) and 48(3) which he still holds.'
    src_list = src.split(' ')
    tar_list = tar_before.split(' ')
    input_list = src_list+tar_list

    inference_single(tokenizer, model, input_list, crf)
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', default=False, action='store_true', help='use fp16 training')
    parser.add_argument('--model_checkpoint', default='bert-baes-cased', help="pretrained name of model")
    parser.add_argument('--model_dir', default='./saved_models/', help="Directory containing config.json of model")
    parser.add_argument('--crf', default=False, action='store_true', help='use crf')
    parser.add_argument('--max_length', type=int, default=512, help="max length of tokens")
    main(parser)