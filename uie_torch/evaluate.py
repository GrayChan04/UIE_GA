# Copyright (c) 2022 Heiheiyoyo. All Rights Reserved.
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

from model import UIE
import argparse
from functools import partial

import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

# adding DDP part
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils import IEMapDataset, SpanEvaluator, IEDataset, convert_example, get_relation_type_dict, logger, tqdm, unify_prompt_name


@torch.no_grad()
# evaluate DDP(whj)
def evaluate(model, metric, data_loader, device='gpu', local_rank=-1, loss_fn=None, show_bar=True):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`torch.nn.Module`): A model to classify texts.
        metric(obj:`Metric`): The evaluation metric.
        data_loader(obj:`torch.utils.data.DataLoader`): The dataset loader which generates batches.
    """
    return_loss = False
    if loss_fn is not None:
        return_loss = True
    model.eval()
    metric.reset()
    loss_list = []
    loss_sum = 0
    loss_num = 0
    if show_bar:
        data_loader = tqdm(
            data_loader, desc="Evaluating", unit='batch')
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
        
        # original gpu
        # if device == 'gpu':
        #     input_ids = input_ids.cuda()
        #     token_type_ids = token_type_ids.cuda()
        #     att_mask = att_mask.cuda()

        # adding DDP device setup (whj)
        if device == 'gpu':
            input_ids = input_ids.to(local_rank)
            token_type_ids = token_type_ids.to(local_rank)
            att_mask = att_mask.to(local_rank)

        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=att_mask)
        start_prob, end_prob = outputs[0], outputs[1]
        if device == 'gpu':
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
        start_ids = start_ids.type(torch.float32)
        end_ids = end_ids.type(torch.float32)

        if return_loss:
            # Calculate loss
            loss_start = loss_fn(start_prob, start_ids)
            loss_end = loss_fn(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss = float(loss)
            loss_list.append(loss)
            loss_sum += loss
            loss_num += 1
            if show_bar:
                data_loader.set_postfix(
                    {
                        'dev loss': f'{loss_sum / loss_num:.5f}'
                    }
                )

        # Calcalate metric
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob,
                                                           start_ids, end_ids)

        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    if return_loss:
        loss_avg = sum(loss_list) / len(loss_list)
        return loss_avg, precision, recall, f1
    else:
        return precision, recall, f1


def do_eval():

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model = UIE.from_pretrained(args.model_path)

    # original gpu
    # if args.device == 'gpu':
    #     model = model.cuda()

    # adding DDP part(whj)
    if args.device == 'gpu':
        # adding DDP initializaton(whj)
        args.local_rank=int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            world_size=torch.cuda.device_count(),
            rank=args.local_rank
        )
        # adding DDP model(whj)
        model = model.to(args.local_rank)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        
    test_ds = IEDataset(args.test_path, tokenizer=tokenizer,
                        max_seq_len=args.max_seq_len)
    
    # origianl dataloader
    # test_data_loader = DataLoader(
    #     test_ds, batch_size=args.batch_size, shuffle=False)
    
    # adding DDP dataloader(whj)
    test_sampler = DistributedSampler(test_ds)
    test_data_loader = DataLoader(
        test_ds, batch_size=args.batch_size, sampler=test_sampler)

    class_dict = {}
    relation_data = []
    if args.debug:
        for data in test_ds.dataset:
            class_name = unify_prompt_name(data['prompt'])
            # Only positive examples are evaluated in debug mode
            if len(data['result_list']) != 0:
                if "的" not in data['prompt']:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data['prompt'], data))
        relation_type_dict = get_relation_type_dict(relation_data)
    else:
        class_dict["all_classes"] = test_ds

    for key in class_dict.keys():
        if args.debug:
            test_ds = IEMapDataset(class_dict[key], tokenizer=tokenizer,
                                   max_seq_len=args.max_seq_len)
        else:
            test_ds = class_dict[key]
        
        # origianl dataloader
        # test_data_loader = DataLoader(
        #     test_ds, batch_size=args.batch_size, shuffle=False)

        # adding DDP dataloader(whj)
        test_data_loader = DataLoader(
            test_ds, batch_size=args.batch_size, sampler=test_sampler)
        
        metric = SpanEvaluator()

        # original evaluate
        # precision, recall, f1 = evaluate(
        #     model, metric, test_data_loader, args.device)
        
        # DDP evaluate (whj)
        precision, recall, f1 = evaluate(
            model, metric, test_data_loader, args.device, args.local_rank)
        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
                    (precision, recall, f1))

    if args.debug and len(relation_type_dict.keys()) != 0:
        for key in relation_type_dict.keys():
            test_ds = IEMapDataset(relation_type_dict[key], tokenizer=tokenizer,
                                   max_seq_len=args.max_seq_len)
            # original dataloader
            # test_data_loader = DataLoader(
            #     test_ds, batch_size=args.batch_size, shuffle=False)

            # DDP dataloader(whj)
            test_data_loader = DataLoader(
                test_ds, batch_size=args.batch_size, sampler=test_sampler)
            
            metric = SpanEvaluator()

            # origianl evaluate
            # precision, recall, f1 = evaluate(
            #     model, metric, test_data_loader, args.device)

            # DDP evaluate(whj)
            precision, recall, f1 = evaluate(
                model, metric, test_data_loader, args.device, args.local_rank)
            
            logger.info("-----------------------------")
            logger.info("Class Name: X的%s" % key)
            logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
                        (precision, recall, f1))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="The path of saved model that you want to load.")
    parser.add_argument("-t", "--test_path", type=str, required=True,
                        help="The path of test set.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("-D", '--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to run model, defaults to gpu.")
    parser.add_argument("--debug", action='store_true',
                        help="Precision, recall and F1 score are calculated for each class separately if this option is enabled.")
    #adding ddp argument(whj)
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help="node rank for distributed training")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
