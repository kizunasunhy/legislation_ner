# legislation_ner
NER for legislation automation  
Using Roberta/Longformer/Bigbird etc.  
for regular NER and Span NER
## Usage
### Requirement
```
PyTorch 1.8 or higher
scikit-learn
tqdm
pandas
transformers
pytorch-crf
```
### Inference Demo

## Dataset
### NER tag
O  
Substitute  
Before-insert  
After-insert  
Omit

This includes the most common amendment types for UK legislations.  
For more information, please refer to the official website of [UK legislation](https://www.legislation.gov.uk/).
## Training
### Preperation

### Start training
For regular NER,  
```
$ python train.py --model_checkpoint 'roberta-base' --fp16
```
For span NER,  
```
$ python train_span.py --model_checkpoint 'roberta-base' --fp16
```
We highly recommend using NVIDIA's Automatic Mixed Precision (AMP) for acceleration.
Install the [APEX](https://github.com/NVIDIA/apex) first and then turn on the "-fp16" option.
## Performance
### Criteria
There are several stantard to evaluate the performance of a multi-class classification model like NER.
First the simplest criteria is global accuracy. If we've got the confusion matrix, 

`global accuracy = confusion_matrix.trace()/confusion_matrix.sum()`

But it doesn't reflect the accuracy of every class's accuracy. And in multi class classification's situation,
micro f1 score and macro f1 score are more frequently used for evaluation.
micro f1 score doesn't distinguish classes, instead it calculates the overall TP (True Positive), FP (False Positive), FN (False Negative):
```
precision = TP/ (TP + FP)
recall = TP/( TP + FN)
micro f1 score = 2 * precision * recall/(precision + recall)
```
While macro f1 score uses the same formula to calculate every class's f1 scores F11, F12, F13,... 
and then average them. In the situation of n classes, macro f1 score is like this:
```
macro f1 score = (F11 + F12 + F13,...)/n
```
In this project, we consider macro f1 score the most, and micro f1 score and global accuray at the same time.

The results after 5 epochs are as follows.
| Model | macro f1 score |
| ------------ | ------------- |
| Roberta-base-batchsize8-lr2e5 | 0.986 |


## In the future
### Auto-amendment
Using pipeline and joint models conbined by seq2seq models.  
