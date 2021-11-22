import transformers
from dataset import encode_tags, NERdataset, SPAN_NERdataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import BertConfig, RobertaConfig
from transformers.optimization import AdamW
from sklearn.metrics import classification_report
from model import BertSpanForNer, RobertaSpanNer
from utils_span import Processor, InputFeature
from utils import define_label
from torch.utils.data import TensorDataset
from collections import Counter
import logging
import numpy as np
import os
import argparse

def convert_examples_to_features(examples, tokenizer, texts_dict, short_label_list, separate_src_tar=True, max_length=512, model_name='bert', mask_label = -100):
    
    if separate_src_tar==True:
        encodings = tokenizer(texts_dict['src'], texts_dict['tar'], is_split_into_words=True, max_length=max_length, 
                              padding=True, truncation='only_second', return_offsets_mapping=True)
    else:
        encodings = tokenizer(texts, is_split_into_words=True, max_length=max_length, 
                              padding=True, truncation=True, return_offsets_mapping=True)
        
    label2id = {label: i for i, label in enumerate(short_label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):

        start_ids = [0] * len(encodings.input_ids[ex_index])
        end_ids = [0] * len(encodings.input_ids[ex_index])
        subjects_id = [0] * len(encodings.input_ids[ex_index])*2
        
        word_ids = encodings.word_ids(ex_index)
        word_idx_dict = {}
        if separate_src_tar==False:
            for index, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in word_idx_dict:
                    continue
                word_idx_dict[word_idx] = index
        else:
            cnt = 0
            length = len(texts_dict['src'][ex_index])
            for index, word_idx in enumerate(word_ids):
                if word_idx is None:
                    cnt += 1
                    continue
                if cnt == 3:
                    word_real_idx = word_idx + length
                else:
                    word_real_idx = word_idx
                if word_real_idx in word_idx_dict:
                    continue
                word_idx_dict[word_real_idx] = index
        #print(word_idx_dict)

        n = 0
        subjects = example.subject
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            #print(start, end)
            if start in word_idx_dict and end in word_idx_dict:
                start_ids[word_idx_dict[start]] = label2id[label]
                end_ids[word_idx_dict[end]] = label2id[label]
                subjects_id[n:n+3] = [label2id[label], word_idx_dict[start], word_idx_dict[end]]
                n+=3

        features.append(InputFeature(input_ids=encodings[ex_index].ids,
                                  input_mask=encodings[ex_index].attention_mask,
                                  segment_ids=encodings[ex_index].type_ids,
                                  start_ids=start_ids,
                                  end_ids=end_ids,
                                  subjects=subjects_id))
    return features
        
#--------------------SKLEARN METRICS-------------------
def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1
    
def compute_metrics(p):
    logits, subjects = p
    start_logits, end_logits =  logits
    #print(start_logits.shape, end_logits.shape, subjects.shape) #(2034, 512, 5) (2034, 512, 5) (2034, 512)
    
    num_examples = start_logits.shape[0]
    origins = []
    founds = []
    rights = []
        
    for index in range(num_examples):
        S = []
        start_pred = np.argmax(start_logits[index], -1)
        end_pred = np.argmax(end_logits[index], -1)
        for i, s_l in enumerate(start_pred):
            if i==0 or s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    S.append([s_l, i, i + j])
                    break
                    
        subject = []
        vec = subjects[index]
        for n in range(len(vec)):
            if n%3 == 0 and vec[n]!=0:
                subject.append([vec[n], vec[n+1], vec[n+2]])
            if n%3 == 0 and vec[n]==0:
                break
        
        origins.extend(subject)
        founds.extend(S)
        rights.extend([pre_entity for pre_entity in S if pre_entity in subject])
        #print(origins, start_pred, end_pred)
        
    id2label = {i: label for i, label in enumerate(short_label_list)}
    class_info = {}
    origin_counter = Counter([id2label[x[0]] for x in origins])
    found_counter = Counter([id2label[x[0]] for x in founds])
    right_counter = Counter([id2label[x[0]] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = compute(origin, found, right)

    return {
        "accuracy": precision,
        "recall": recall,
        "f1": f1,
        'sub-f1':class_info['substitute']['f1'],
        'after-insert-f1':class_info['after-insertions']['f1'],
        'before-insert-f1':class_info['before-insertions']['f1'],
        'omit-f1':class_info['revocation']['f1']
    }


def main(parser):
    
    pargs = parser.parse_args()
    batch_size = pargs.batch_size
    label_list = ['O', 'B-substitute', 'I-substitute', 'B-before-insertions', 'I-before-insertions', 'B-after-insertions', 
              'I-after-insertions', 'B-revocation', 'I-revocation']
    global short_label_list
    short_label_list = ['O', 'substitute', 'before-insertions', 'after-insertions', 'revocation']
    global label_dict_rev
    label_dict, label_dict_rev = define_label(label_list)
    logger = logging.getLogger()
    
    #-------------------------------LOAD DATA-------------------------------
    print('Load data...')
    train_texts = np.load(pargs.data_dir+'train_te.npy', allow_pickle=True)
    val_texts = np.load(pargs.data_dir+'val_te.npy', allow_pickle=True)
    train_tags = np.load(pargs.data_dir+'train_ta.npy', allow_pickle=True)
    val_tags = np.load(pargs.data_dir+'val_ta.npy', allow_pickle=True)
    train_texts = train_texts.tolist()
    val_texts = val_texts.tolist()

    train_texts_dict = {'src':[], 'tar':[]}
    val_texts_dict = {'src':[], 'tar':[]}
    for i in range(len(train_texts)):
        train_texts_dict['src'].append(train_texts[i][0])
        train_texts_dict['tar'].append(train_texts[i][1])
    for i in range(len(val_texts)):
        val_texts_dict['src'].append(val_texts[i][0])
        val_texts_dict['tar'].append(val_texts[i][1])
    
    model_checkpoint = pargs.model_checkpoint
    #"bert-base-cased", 'roberta-base', 'microsoft/mpnet-base', 'allenai/longformer-base-4096', 'nghuyong/ernie-2.0-en'
    #'google/bigbird-roberta-base', 'gpt2', 'microsoft/deberta-base', 'roberta-large'
    model_name = model_checkpoint.split('-')[0].split('/')[-1]

    #-------------------------------LOAD TOKENIZER-------------------------
    print('Load {} tokenizer...'.format(model_name))
    if model_name == 'bert' or model_name == 'ernie' or model_name=='mpnet':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if model_name == 'roberta' or model_name=='longformer' or model_name=='bigbird' or model_name=='gpt2' or model_name=='deberta':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True) #For Roberta and Longformer
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    
    #------------------------------DEFINE DATASET-----------------------
    print('Load dataset...')
    max_length = pargs.max_length
    processor = Processor(label_dict, label_dict_rev)
    train_examples = processor.get_train_examples(train_texts, train_tags)
    val_examples = processor.get_val_examples(val_texts, val_tags)
    train_features = convert_examples_to_features(train_examples,
                                                    tokenizer,
                                                    train_texts_dict,
                                                    short_label_list,
                                                    max_length=max_length,
                                                    model_name=model_name
                                                    )

    val_features = convert_examples_to_features(val_examples,
                                                    tokenizer,
                                                    val_texts_dict,
                                                    short_label_list,
                                                    max_length=max_length,
                                                    model_name=model_name
                                                    )
    train_dataset = SPAN_NERdataset(train_features)
    val_dataset = SPAN_NERdataset(val_features)

    #---------------------------DEFINE MODEL & OPTIMIZER-------------------
    print('Load {} model...'.format(model_name))
    num_labels = len(short_label_list)
    if model_name == 'bert':
        config = BertConfig.from_pretrained(model_checkpoint,num_labels=num_labels)
    if model_name == 'roberta':
        config = RobertaConfig.from_pretrained(model_checkpoint,num_labels=num_labels)
    #config.soft_label = True
    config.loss_type = pargs.loss_type
    #model = BertSpanForNer.from_pretrained(model_checkpoint, config=config)
    model = RobertaSpanNer.from_pretrained(model_checkpoint, config=config)

    #------------------------------DEFINE TRAINER---------------------------
    m_name = model_checkpoint.split("/")[-1]
    output_dir = pargs.model_dir + f"{m_name}-finetuned-span"
    args = TrainingArguments(
        output_dir,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        save_strategy ='epoch',
        load_best_model_at_end=True,
        fp16=True,
        fp16_opt_level='O1',
        weight_decay=0.01,
        push_to_hub=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics)
    
    if pargs.continue_train:
        trainer.train(pargs.continue_training_path)
    else:
        trainer.train()
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', default=False, action='store_true', help='use fp16 training')
    parser.add_argument('--data_dir', default='./data/', help="Directory containing config.json of data")
    parser.add_argument('--model_checkpoint', default='bert-base-cased', help="pretrained name of model")
    parser.add_argument('--model_dir', default='./saved_models/', help="Directory containing config.json of model")
    parser.add_argument('--patience', type=int, default=2, help="Patience if macro f1 score is not increasing")
    parser.add_argument('--continue_train', default=False, action='store_true', help="Continue training.")
    parser.add_argument('--continue_train_path', help="Continue training checkpoint")
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--lr_schedule', default=False, action='store_true', help='Using learning rate scheduler')
    parser.add_argument('--loss_type', default='ce', help="loss calculating method, including 'ce', 'focal', 'lsr', default=cross entropy")
    parser.add_argument('--batch_size', type=int, default=8, help='learning rate')
    parser.add_argument('--max_length', type=int, default=512, help="max length of tokens")
    main(parser)