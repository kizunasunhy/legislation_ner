## PYTORCH CODE
import torch
import numpy as np

def encode_tags(tags, texts, encodings, model_name='bert', label_all_tokens=False, mask_label = -100):
    #Roberta 或者其他使用roberta的tokenizer的model都不能用offset_mapping
    #因为没法区分(1, 1), (1, 8)这种情况
    #这种情况即可以是'G(', 'express'
    #也可以是'G(', 'Grelating'
        
    encoded_labels = []
    assert len(tags)==len(encodings.offset_mapping)
    
    for i in range(len(tags)):
        doc_labels = tags[i]
        doc_offset = encodings.offset_mapping[i]
        
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * mask_label
            
        arr_offset = np.array(doc_offset)
 
        if label_all_tokens == False:
            if model_name == 'bert' or model_name == 'ernie' or model_name=='mpnet':
                # set labels whose first offset position is 0 and the second is not 0
                l = len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)])
                doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels[:l]
                doc_enc_labels[0] = 0
                encoded_labels.append(doc_enc_labels.tolist())
            if model_name=='roberta' or model_name=='longformer' or model_name=='bigbird' or model_name=='gpt2' or model_name=='deberta':
                res=[]
                ress = ''
                word_ids = encodings.word_ids(i)
                length = len(texts[i][0]) #第一句的长度
                cnt = 0
                for j in word_ids:
                    if j is None:
                        cnt += 1
                        res.append(mask_label)
                        continue
                    if cnt == 3:
                        jj = j + length
                    else:
                        jj = j
                    if jj==ress:
                        res.append(mask_label)
                        continue
                    res.append(doc_labels[jj])
                    ress = jj
                encoded_labels.append(res)  
        else:
            word_ids = encodings.word_ids(i)
            res = [mask_label if j is None else doc_labels[j] for j in word_ids]
            res[0] = 0
            encoded_labels.append(res)
        
    return encoded_labels


class NERdataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SPAN_NERdataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features
        
    def __getitem__(self, idx):
        
        item = {}
        item['input_ids'] = torch.tensor(self.features[idx].input_ids)
        item['token_type_ids'] = torch.tensor(self.features[idx].segment_ids)
        item['attention_mask'] = torch.tensor(self.features[idx].input_mask)
        item['start_positions'] = torch.tensor(self.features[idx].start_ids)
        item['end_positions'] = torch.tensor(self.features[idx].end_ids)
        item['labels'] = torch.tensor(self.features[idx].subjects)
        return item

    def __len__(self):
        return len(self.features)