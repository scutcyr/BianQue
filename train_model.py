# coding=utf-8
# Copyright 2023 Research Center of Body Data Science from South China University of Technology. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# File: train_model.py
# Description: The training model code
# Repository: https://github.com/scutcyr
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2023/03/14


import re
import os
import copy
import json
import glob
import time
import torch
import random
import logging
import shutil
import numpy as np
import pandas as pd
from pprint import pformat
from tqdm import tqdm, trange
from typing import Dict, List, Tuple

from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW as torch_AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# 优化器
from transformers.optimization import (AdamW, Adafactor)
# 学习率模式
from transformers.optimization import (get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, 
    get_constant_schedule, get_cosine_schedule_with_warmup, get_adafactor_schedule)

# 导入数据类
from utils import PromptDataSetClass
from train_model_args import parser # 导入模型所需参数

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__file__)


def setup_seed(seed, n_gpu):
    ''' 设置随机种子 '''
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def average_distributed_scalar(scalar, args):
    ''' Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. '''
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ''' 对所有的checkpoints进行排序 '''
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    ''' 删除多出的checkpoint '''
    if not args.save_total_limit or args.save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, model, tokenizer, train_dataset, eval_dataset=None):
    '''
    Training the model
    '''
    logger.info("***** Preparing for training *****")
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    def collate_train(examples):
        #print(examples)
        source_ids = list(map(lambda x: x[0], examples))
        source_mask = list(map(lambda x: x[1], examples))
        target_ids = list(map(lambda x: x[2], examples))
        target_mask = list(map(lambda x: x[3], examples))

        source_ids = torch.vstack(source_ids)
        source_mask = torch.vstack(source_mask)
        target_ids = torch.vstack(target_ids)
        target_mask = torch.vstack(target_mask)

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
    
    # RandomSampler：随机采样；DistributedSampler：分布式采样
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 or args.model_parallel else DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_train)
    
    # 计算总的训练step数目：max_steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # 设置需要进行L2正则化的参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}
        ]

    # 训练时的优化器选择
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.optimizer == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "Adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0, scale_parameter=True, relative_step=True, warmup_init=True)
    elif args.optimizer == "Adafactor-srwf":
        # 此种情形无需scheduler
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, eps=(1e-30, 1e-3), clip_threshold=1.0, scale_parameter=False, relative_step=False, warmup_init=False)

    # 学习率模式选择，2022/08/30 新增 by Yirong Chen
    if args.scheduler == "get_linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_ratio*t_total), num_training_steps=t_total)
    elif args.scheduler == "get_constant_schedule_with_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_ratio*t_total))
    elif args.scheduler == "get_constant_schedule":
        scheduler = get_constant_schedule(optimizer)
    elif args.scheduler == "get_cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up_ratio*t_total), num_training_steps=t_total, num_cycles=0.5)
    elif args.scheduler == "get_adafactor_schedule":
        # 配合Adafactor进行使用
        # min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
        # KeyError: 'step'
        scheduler = get_adafactor_schedule(optimizer, initial_lr=args.learning_rate)
    elif args.scheduler == "no_schedule":
        logger.info("***** Not use any scheduler!!! *****")

    # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        if args.scheduler != "no_schedule":
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.autocast: # 混合精度训练
        # 参考: https://pytorch.org/docs/1.9.0/amp.html?highlight=torch%20cuda%20amp%20gradscaler
        #       https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
        scaler = GradScaler()  # pytorch版本要求：1.6+

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and args.local_rank == -1 and not args.model_parallel:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1 and not args.model_parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                          find_unused_parameters=not args.not_find_unused_parameters)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")


    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    setup_seed(args.seed, args.n_gpu)

    
    for epoch_i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        time1=time.time()
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # 输入数据规范化
            if args.model_parallel:
                y = batch["target_ids"].to(model.encoder.first_device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
                lm_labels = y[:, 1:].clone().detach() # target, for second to end.e.g."好吗？<EOS>"
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
                ids = batch["source_ids"].to(model.encoder.first_device, dtype=torch.long) # input. e.g. "how are you?"
                mask = batch["source_mask"].to(model.encoder.first_device, dtype=torch.long)
            else:
                y = batch["target_ids"].to(args.device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
                lm_labels = y[:, 1:].clone().detach() # target, for second to end.e.g."好吗？<EOS>"
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
                ids = batch["source_ids"].to(args.device, dtype=torch.long) # input. e.g. "how are you?"
                mask = batch["source_mask"].to(args.device, dtype=torch.long)

            # 设置模型为train()模式
            model.train()
            # 模型前向计算

            if args.autocast:
                # 开启了混合精度训练
                with autocast():
                    outputs = model(
                        input_ids=ids,
                        attention_mask=mask,
                        decoder_input_ids=y_ids,
                        labels=lm_labels)
                    
                    # 获得模型的loss
                    loss = outputs[0]
                    # 对于DDP训练模式，需要将loss求平均
                    if args.n_gpu > 1:
                        loss = loss.mean()
                    # 当设置的gradient_accumulation_steps大于1时，对loss进行loss除以gradient_accumulation_steps
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                
                # 梯度后向计算更新参数
                scaler.scale(loss).backward()
                #scaler.scale(loss).backward(retain_graph=args.retain_graph) 
                # retain_graph here is unrelated to amp, it's present because in this both backward() calls share some sections of graph.
                
                # 记录累积loss
                tr_loss += loss.item()
                # 梯度裁剪、优化器和学习率模式更新
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # 参考：https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual
                    parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1


            else:
                outputs = model(
                            input_ids=ids,
                            attention_mask=mask,
                            decoder_input_ids=y_ids,
                            labels=lm_labels)
                # 获得模型的loss
                loss = outputs[0]
                # 对于DDP训练模式，需要将loss求平均
                if args.n_gpu > 1:
                    loss = loss.mean()
                # 当设置的gradient_accumulation_steps大于1时，对loss进行loss除以gradient_accumulation_steps
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # 梯度后向计算更新参数
                loss.backward()
                # 记录累积loss
                tr_loss += loss.item()
                # 梯度裁剪、优化器和学习率模式更新
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.max_grad_norm)
                    optimizer.step()
                    if args.scheduler != "no_schedule":
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1



            # 打印日志到终端以及保存到Tensorboard文件
            if (args.local_rank in [-1, 0] and args.log_steps > 0 and (global_step % args.log_steps == 0) and (global_step > 0 )):
                if args.scheduler != "no_schedule":
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train/loss", (tr_loss - logging_loss) / args.log_steps, global_step)
                    time2=time.time()
                    epoch_iterator.set_postfix(loss=f'{((tr_loss - logging_loss) / args.log_steps):.4f}', time_per_step=f'{(float(time2-time1)/float(step+0.0001)):.4f}')
                    logging_loss = tr_loss

        # 每个 epoch 保存一次模型
        if args.local_rank in [-1, 0]:
            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, epoch_i))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)
            if args.save_optimizer_and_scheduler:
                # 文件太大了，非必要不保存，需要指定--save_optimizer_and_scheduler才保存
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                if args.scheduler != "no_schedule":
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break      
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss/global_step

def main():
    '''
    训练的主函数入口
    
    '''
    # Get the arguments for training model
    args = parser.parse_args()

    print(args)


    # 导入模型类
    if args.autocast:
        # 与原来的transformers提供的区别是，在T5ForConditionalGeneration的forward函数增加了@autocast()注解
        # 参考：https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
        from models.t5 import T5Tokenizer, T5ForConditionalGeneration
    else:
        from transformers import T5Tokenizer, T5ForConditionalGeneration

    # Setup args.local_rank and args.world_size
    if args.no_cuda or args.model_parallel:
        args.local_rank = -1
        args.world_size = 1
    else:
        args.world_size = 1 # 默认值
        if "LOCAL_RANK" in os.environ: # 需要使用torchrun才有！
            args.local_rank = int(os.environ["LOCAL_RANK"])
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"]) # 节点数*每个节点地方任务数

    # Load model from checkpoints for continue training
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir and not args.should_continue):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # 数据并行DDP训练方式
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                        datefmt="%m/%d/%Y %H:%M:%S", 
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    setup_seed(args.seed, args.n_gpu)

    # Initialize the model and tokenizer
    # 根据模型类型确定模型类以及tokenizer
    if args.model_type == 't5':
        model_class, tokenizer_class = T5ForConditionalGeneration, T5Tokenizer
    '''
    # Add new model class here
    elif args.model_type == "XXX":
        model_class, tokenizer_class = ModelClassName, TokenizerClassName
    '''

    # 加载tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
                special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")
    
    # Load model from pretrained    
    model = model_class.from_pretrained(args.model_name_or_path)

    # Set model parallel training
    # 本部分用于配置模型分配不同的层到不同的GPU卡上进行训练，通常参数达到上十亿的模型才需要
    if args.model_parallel and not args.no_cuda and torch.cuda.is_available():
        logger.info("parallelizing...")
        model.parallelize() # 自动计算模型层数，将模型切分到不同的显卡
    else:
        logger.info("put model to GPU")
        model.to(args.device)

    #if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache


    if args.no_cuda:
        # 辅助调试代码
        print(model)
        print(model.config)

    logger.info("Training/evaluation parameters %s", args)


    # Set args.train_batch_size and args.eval_batch_size
    if args.model_parallel: # 模型流水线并行
        args.train_batch_size = args.per_gpu_train_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size
    else: # 分布式训练
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)



    # Load Dataset
    logger.info("Loading Dataset...")
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=args.dataset_sample_frac)
    print("df.head:",df.head(n=5))
    print("df.shape:",df.shape)

    df = df[[args.dataset_input_column_name, args.dataset_target_column_name]] # 输入和输出的列名

    train_dataset = df.sample(frac=args.train_radio_of_dataset, random_state=args.seed) # 采样得到训练集
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True) # 采样得到验证集
    train_dataset = train_dataset.reset_index(drop=True)

    logger.info(f"FULL Dataset: {df.shape}")
    logger.info(f"TRAIN Dataset: {train_dataset.shape}")
    logger.info(f"TEST Dataset: {val_dataset.shape}\n")

    training_set = PromptDataSetClass(
        train_dataset,
        tokenizer,
        args.max_source_text_length,
        args.max_target_text_length,
        args.dataset_input_column_name,
        args.dataset_target_column_name,
    )
    val_set = PromptDataSetClass(
        val_dataset,
        tokenizer,
        args.max_source_text_length,
        args.max_target_text_length,
        args.dataset_input_column_name,
        args.dataset_target_column_name,
    )

    # Training the model
    if not args.no_train: # 如果不指定--no_train，则进行模型训练
        logger.info("***************Training***************")
        global_step, train_loss = train(args=args, model=model, tokenizer=tokenizer, train_dataset=training_set, eval_dataset=val_set)
        logger.info(" global_step = %s, average loss = %s", global_step, train_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            if args.local_rank in [-1, 0]:  
                os.makedirs(args.output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "training_args.json"),'w',encoding='utf-8') as json_file:
                json.dump(pformat(args),json_file,ensure_ascii=False)
        logger.info("Finishing Train and save model checkpoint!!!")


if __name__ == '__main__':
    main()
