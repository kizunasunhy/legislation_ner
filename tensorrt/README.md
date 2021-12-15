## Usage
```
python3 ./inference.py -e ./span_ner.trt -b 1 -s 512 -w 50 
```
## Transfer to ONNX

First, define your own model, in this case,
```
import torch.nn as nn
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RobertaSpanNer.from_pretrained('./roberta-base-finetuned-span/checkpoint-4380', config=config)
    def forward(self, input_ids, attention_mask):
        start_logit, end_logit = self.model(input_ids, attention_mask)
        return start_logit, end_logit
    
cu_model = CustomModel()
```
And then make some random inputs to the model when exporting ONNX model.  
Be careful! here in this model we have 2 inputs which are "input_ids" and "attention_mask", and "attention_mask" will always be 0 or 1. for "input_ids", we can easily set the range and create a 1* 512 tensor.  
However, for "attention_mask", it will be better if you order those 0 and 1s randomly also, otherwise when you create a tensorrt engine, if you increase the batch size and pad some 0s after the 1s in attention mask, you will encounter error!
```
# Some standard imports
import io
import numpy as np

#
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init

cu_model.eval()

# Input to the model
x1 = torch.randint(1, 100, (batch_size, 512), dtype=torch.int32)
x2 = torch.randint(0, 2, (batch_size, 512), dtype=torch.int32) #very important!
torch_out = cu_model(x1, x2)

# Export the model
torch.onnx.export(cu_model,               # model being run
                  (x1, x2),                         # model input (or a tuple for multiple inputs)
                  "span_ner.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input_ids', 'attention_mask'],   # the model's input names
                  output_names = ['start_logit', 'end_logit'], # the model's output names
                  dynamic_axes={'input_ids' : {0 : 'batch_size', 1:'sequence'},    # variable length axes
                                'attention_mask' : {0 : 'batch_size', 1:'sequence'},
                                'start_logit' : {0 : 'batch_size',1:'sequence'},
                                'end_logit' : {0 : 'batch_size',1:'sequence'}}
                  )

```
## Transfer to tensorrt engine
For common transition,  
```
trtexec --onnx=span_ner.onnx --saveEngine=span_ner.trt --explicitBatch --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:1x512,attention_mask:1x512 --maxShapes=input_ids:1x512,attention_mask:1x512 --workspace=1024
```
For fp16 or int8 inference,  
```
trtexec --onnx=span_ner.onnx --saveEngine=span_ner_int8.trt --explicitBatch --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:1x512,attention_mask:1x512 --maxShapes=input_ids:1x512,attention_mask:1x512 --workspace=1024 --fp16 --int8 --best
```
Be careful! The default value of workspace is 16(MB), which is a little too small.  
If you don't set the size of workspace, you will meet core dumped error when setting bigger input shapes due to the limit of GPU memory.

