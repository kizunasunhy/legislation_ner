import transformers
from datasets import load_dataset, load_metric
from dataset import encode_tags, NERdataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers.optimization import AdamW
from sklearn.metrics import classification_report
from model import Bert_CRF, Roberta_CRF
from utils import define_label

import numpy as np
import os
import argparse


def define_optimizer(model, model_name, lr, crf_lr):
    
    crf_learning_rate = crf_lr
    learning_rate = lr
    no_decay = ["bias", "LayerNorm.weight"]
    if model_name == 'bert':
        bert_param_optimizer = list(model.bert.named_parameters())
    if model_name == 'roberta':
        bert_param_optimizer = list(model.roberta.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())

    optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr': learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': 2e-5},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr': crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr': crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': crf_learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)
    return optimizer
        
'''
#---------------------SEQEVAL METRICS-------------------
metric = load_metric("./seqeval_metrics.py")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
'''
#--------------------SKLEARN METRICS-------------------
def compute_sk_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_dict_rev[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_dict_rev[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_predictions = np.concatenate(true_predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    #target_names = ['class{}'.format(str(i)) for i in range(5)]
    sklearn_dict = classification_report(flat_true_labels, flat_predictions, output_dict=True)

    return {
        "accuracy": sklearn_dict['accuracy'],
        "macro f1": sklearn_dict['macro avg']['f1-score'],
        "weighted f1": sklearn_dict['weighted avg']['f1-score'],
    }

#--------------------SKLEARN CRF METRICS-------------------
def compute_crf_metrics(p):
    logits, labels = p
    logits = torch.from_numpy(logits)
    predictions = model.crf.decode(logits.cuda()) 
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_dict_rev[p] for (p, l) in zip(prediction, label) if l != -1]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_dict_rev[l] for (p, l) in zip(prediction, label) if l != -1]
        for prediction, label in zip(predictions, labels)
    ]

    flat_predictions = np.concatenate(true_predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    #target_names = ['class{}'.format(str(i)) for i in range(5)]
    sklearn_dict = classification_report(flat_true_labels, flat_predictions, output_dict=True)

    return {
        "accuracy": sklearn_dict['accuracy'],
        "macro f1": sklearn_dict['macro avg']['f1-score'],
        "weighted f1": sklearn_dict['weighted avg']['f1-score'],
    }
    


def main(parser):
    
    pargs = parser.parse_args()
    batch_size = pargs.batch_size
    crf = pargs.crf
    label_list = ['O', 'B-substitute', 'I-substitute', 'B-before-insertions', 'I-before-insertions', 'B-after-insertions', 
              'I-after-insertions', 'B-revocation', 'I-revocation']
    global label_dict_rev
    label_dict, label_dict_rev = define_label(label_list)
    
    #-------------------------------LOAD DATA-------------------------------
    print('Load data...')
    train_texts = np.load(pargs.data_dir+'train_te.npy', allow_pickle=True)
    val_texts = np.load(pargs.data_dir+'val_te.npy', allow_pickle=True)
    train_tags = np.load(pargs.data_dir+'train_ta.npy', allow_pickle=True)
    val_tags = np.load(pargs.data_dir+'val_ta.npy', allow_pickle=True)
    train_texts = train_texts.tolist()
    val_texts = val_texts.tolist()

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
    if model_name == 'gpt2':
        ADD_SPECIAL_TOKENS = True
        num_added_tokens = tokenizer.add_special_tokens({'pad_token':'<PAD>'})
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    max_length = pargs.max_length
    train_encodings = tokenizer(train_texts, is_split_into_words=True, max_length=max_length, padding=True, truncation=True,
                                return_offsets_mapping=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, 
                              return_offsets_mapping=True, max_length=max_length, padding=True, truncation=True)

    #------------------------------DEFINE DATASET-----------------------
    print('Load dataset...')
    if crf == True:
        label_all_tokens = True
        mask_label = -1
    else:
        label_all_tokens = False
        mask_label = -100
    train_labels = encode_tags(train_tags, train_encodings, model_name, label_all_tokens, mask_label = mask_label)
    val_labels = encode_tags(val_tags, val_encodings, model_name, label_all_tokens, mask_label = mask_label)
    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = NERdataset(train_encodings, train_labels)
    val_dataset = NERdataset(val_encodings, val_labels)

    #---------------------------DEFINE MODEL & OPTIMIZER-------------------
    print('Load {} model...'.format(model_name))
    if crf == True:
        if model_name == bert:
            model = Bert_CRF.from_pretrained(model_checkpoint, num_labels=len(label_list))
        if model_name == roberta:
            model = Roberta_CRF.from_pretrained(model_checkpoint, num_labels=len(label_list))  
        optimizer = define_optimizer(model, model_name, pargs.lr, pargs.crf_lr)
        optimizers = (optimizer, None)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
        if model_name == 'gpt2':
            model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + num_added_tokens)
        optimizers = (None, None)

    #------------------------------DEFINE TRAINER---------------------------
    m_name = model_checkpoint.split("/")[-1]
    output_dir = pargs.model_dir + f"{m_name}-finetuned-ner"
    args = TrainingArguments(
        output_dir,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        #lr_scheduler_type='linear',
        #warmup_steps=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        save_strategy ='epoch',
        load_best_model_at_end=True,
        fp16=True,
        fp16_opt_level='O1',
        logging_dir='./',
        logging_strategy='epoch',
        weight_decay=0.01,
        push_to_hub=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    if crf:
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_crf_metrics,
            optimizers = optimizers)
    else:
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_sk_metrics)
    
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
    parser.add_argument('--crf', default=False, action='store_true', help='use crf')
    parser.add_argument('--crf_lr', type=float, default=2e-2, help='learning rate')
    parser.add_argument('--lr_schedule', default=False, action='store_true', help='Using learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='learning rate')
    parser.add_argument('--max_length', type=int, default=512, help="max length of tokens")
    main(parser)