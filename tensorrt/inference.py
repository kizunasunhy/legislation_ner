#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Written by Sun Hongyang.
This script uses a TensorRT engine of customized span NER model
which can predict start and end positions for 1 batch in less than 5ms.
"""

import time
import ctypes
import argparse
import collections
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from transformers import AutoTokenizer
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class DeviceBuffer(object):
    def __init__(self, shape, dtype=trt.int32):
        self.buf = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

    def binding(self):
        return int(self.buf)

    def free(self):
        self.buf.free()

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--engine',
            help='Path to TensorRT engine')
    parser.add_argument("-b", "--batch-size", default=1, help="Batch size for inference.", type=int)
    parser.add_argument('-s', '--sequence-length',
            help='The sequence length to use. Defaults to 128',
            default=128, type=int)
    parser.add_argument('-w', '--warm-up-runs', default=10, help='Number of iterations to run prior to benchmarking.', type=int)
    args, _ = parser.parse_known_args()
    return args

def decoding(start_pos, end_pos, encodings, input_list):
    word_ids = encodings.word_ids()
    length = len(input_list[0])
    cnt = 0
    for i, ids in enumerate(word_ids):
        if word_ids[i] is None:
            cnt += 1
            continue
        if cnt == 3:
            word_ids[i] += length    
    #print(word_ids)      
    start_word_dict = {}
    end_word_dict = {}
    for n in range(len(start_pos)):
        if word_ids[n] in start_word_dict or word_ids[n] is None:
            continue
        start_word_dict[word_ids[n]] = int(start_pos[n])
    for n in range(len(end_pos)):
        if word_ids[n] in end_word_dict or word_ids[n] is None:
            continue
        end_word_dict[word_ids[n]] = int(end_pos[n])  
        
    #-----------------------Print Results-----------------------
    input_list = input_list[0] + input_list[1]
    for i, enc in enumerate(input_list):
        print(i, enc, "\t", start_word_dict.get(i, 'NaN'), end_word_dict.get(i, 'NaN'))
    print('===========================================')
    return

if __name__ == '__main__':
    args = parse_args()

    src = 'in sub-paragraph (6)(a) for the words after “under paragraph (1) or regulation 7(1)” insert “under paragraph (1), regulation 7(1) or regulation 8(1)”;'
    tar_before = 'a prescribed sum is payable by that licensee in respect of a licence under paragraph (1) or regulation 7(1);'
    #src = 'In subsection (1A) , for the words from the beginning to “under section 240 or 240A”, substitute “section 240ZA includes”.'
    #tar_before = 'In subsection (1) the reference to a direction under section 240 or 240A includes a direction under section 246 of the Armed Forces Act 2006.'
    input_list = [src.split(' '), tar_before.split(' ')]

    model_checkpoint = 'roberta-base'
    short_label_list = ['O', 'substitute', 'before-insertions', 'after-insertions', 'revocation']
    num_labels = len(short_label_list)
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    max_seq_length = args.sequence_length

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True) #For Roberta and Longformer
    test_encodings = tokenizer(input_list[0], input_list[1], is_split_into_words=True,
                               return_offsets_mapping=True, padding="max_length", max_length=max_seq_length, truncation='only_second')
    input_ids = np.asarray(test_encodings["input_ids"], dtype=np.int32, order=None)
    attention_mask = np.asarray(test_encodings["attention_mask"], dtype=np.int32, order=None)

    # Import necessary plugins for BERT TensorRT
    handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    if not handle:
        raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        # Allocate buffers large enough to store the largest batch size
        max_input_shape = (args.batch_size, max_seq_length)
        max_output_shape = (args.batch_size, max_seq_length, 5) #5 classes for span ner
        buffers = [
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_output_shape),
            DeviceBuffer(max_output_shape)
        ]

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        
        # select engine profile
        selected_profile = -1
        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
        for idx in range(engine.num_optimization_profiles):
            profile_shape = engine.get_profile_shape(profile_index = idx, binding = idx * num_binding_per_profile)
            #idx=0 profile_shape=[(1, 1), (1, 256), (1, 256)]
            if profile_shape[0][0] <= args.batch_size and profile_shape[2][0] >= args.batch_size and profile_shape[0][1] <= max_seq_length and profile_shape[2][1] >= max_seq_length:
                selected_profile = idx
                break
        if selected_profile == -1:
            raise RuntimeError("Could not find any profile that can run batch size {}.".format(args.batch_size))
        context.set_optimization_profile_async(selected_profile, stream.handle)
        binding_idx_offset = selected_profile * num_binding_per_profile
        #context.set_optimization_profile_async(selected_profile, stream.handle)

        # Each profile has unique bindings
        bindings = [0] * binding_idx_offset + [buf.binding() for buf in buffers]
        # print(bindings)

        # Specify input shapes. These must be within the min/max bounds of the active profile
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        input_shape = (args.batch_size, max_seq_length)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
        shapes = {
                "input_ids": (args.batch_size, max_seq_length),
                "attention_mask": (args.batch_size, max_seq_length)
                }
        for binding, shape in shapes.items():
            context.set_binding_shape(engine[binding] + binding_idx_offset, shape)
        assert context.all_binding_shapes_specified

        # Allocate device memory for inputs.
        d_inputs = [buffers[0].buf, buffers[1].buf]
        #d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(2)]
        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output1 = cuda.pagelocked_empty(tuple(max_output_shape), dtype=np.float32)
        h_output2 = cuda.pagelocked_empty(tuple(max_output_shape), dtype=np.float32)
        d_output1 = buffers[2].buf
        d_output2 = buffers[3].buf

        # Warmup
        for _ in range(args.warm_up_runs):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()

        def inference(input_ids, attention_mask):
            global h_output1, h_output2

            _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                    "NetworkOutput",
                    ["start_logits", "end_logits"])
            networkOutputs = []

            eval_time_elapsed = 0
            eval_start_time = time.time()
            # Copy inputs
            input_ids_batch = np.repeat(np.expand_dims(input_ids, 0), args.batch_size, axis=0)
            attention_mask_batch = np.repeat(np.expand_dims(attention_mask, 0), args.batch_size, axis=0)
            # print(input_ids_batch.ravel(), attention_mask_batch.ravel())
            input_ids_h = cuda.register_host_memory(np.ascontiguousarray(input_ids_batch.ravel()))
            attention_mask_h = cuda.register_host_memory(np.ascontiguousarray(attention_mask_batch.ravel()))
            
            cuda.memcpy_htod_async(d_inputs[0], input_ids_h, stream)
            cuda.memcpy_htod_async(d_inputs[1], attention_mask_h, stream)

            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Synchronize the stream
            stream.synchronize()
            
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
            cuda.memcpy_dtoh_async(h_output2, d_output2, stream)
            stream.synchronize()
            eval_time_elapsed += (time.time() - eval_start_time)

            # Only retrieve and post-process the first batch
            start_logits = np.array(h_output1[0])
            end_logits = np.array(h_output2[0])
            # print(start_logits.shape)
            start_position = np.argmax(start_logits, axis=1)
            end_position = np.argmax(end_logits, axis=1)
            networkOutputs.append(_NetworkOutput(
                    start_logits = np.array(h_output1[0]),
                    end_logits = np.array(h_output2[0]),
                    ))

            return start_position, end_position, eval_time_elapsed

        start_position, end_position, eval_time_elapsed = inference(input_ids, attention_mask)
        [b.free() for b in buffers]

    # print(start_position, end_position)
    print("---------------------------------")
    print("Running inference in {} batch(es) and cost {:.2} ms".format(args.batch_size, eval_time_elapsed*1000))
    print("---------------------------------")
    print("src: {}".format(src), '\n')
    print("tar_before: {}".format(tar_before))
    decoding(start_position, end_position, test_encodings, input_list)
