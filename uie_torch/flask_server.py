# coding=utf-8
# Copyright (C) xxx team - All Rights Reserved
#
# @Version:   3.11.4
# @Software:  PyCharm
# @FileName:  flask_server.py
# @CTime:     2023/10/19 11:16   
# @Author:    xxx
# @Email:     xxx
# @UTime:     2023/10/19 11:16
#
# @Description:
#     xxx
#     xxx
#
import json
import logging
from typing import List, Dict
import onnxruntime
from flask import Flask, request
from uie_predictor import parse_args, UIEPredictor
from export_model import export_onnx
from model import UIE, UIEM
import numpy as np
import os
from onnxruntime import InferenceSession, SessionOptions

logger = logging.getLogger(__name__)


def model_func(text):
    args = parse_args()
    args.engine = 'pytorch'
    args.device = 'cpu'
    args.position_prob = 0.5
    args.max_seq_len = 512
    args.batch_size = 64
    args.use_fp16 = False
    args.split_sentence = False
    args.schema = {'人-报警人/报案人': ['物-手机号码', '物-身份证号', '物-民族']}
    args.schema_lang = "zh"
    uie = UIEPredictor(model=args.model, task_path='./export', schema_lang=args.schema_lang, schema=args.schema,
                       engine='onnx', device=args.device,
                       position_prob=args.position_prob, max_seq_len=args.max_seq_len, batch_size=2,
                       split_sentence=False, use_fp16=args.use_fp16)
    return uie(text)
    
def convert_np_float32(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_np_float32(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_np_float32(value) for key, value in obj.items()}
    return obj

app = Flask(__name__)
@app.route('/uie', methods=[ "GET"])
def universalEntityExtract():
    if request.method == 'GET':
        data = request.get_json()
        text = data['text']
        uie_result = model_func(text)
        # float32类型的数据无法进行JSON序列化，转成float
        for entity_type in uie_result:
            for entity in entity_type.values():
                for item in entity:
                    if 'probability' in item:
                        item['probability'] = float(item['probability'])

        uie_dict = {'entity_list': uie_result}
        result = json.dumps(convert_np_float32(uie_dict), ensure_ascii=False)
        print(result)
    return result

if __name__ == '__main__':
    app.run('0.0.0.0', port=6020)
