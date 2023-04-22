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


# File: train_model_args.py
# Description: The training model code for SCUTChatLM
# Repository: https://github.com/scutcyr
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2023/03/14
# Usage:
#    from train_model_args import parser
#    args = parser.parse_args()


import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "--model_type",
    default="t5",
    type=str,
    choices=['t5'],
    help="The model architecture to be trained or fine-tuned.",
)



# model_parallel: 设置模型是否并行，也就是将一个超大模型放在多张GPU上
parser.add_argument(
    "--model_parallel",
    action="store_true",
    help="Set model_parallel=True",
)
# 预训练模型路径或名称或者初始化配置路径
parser.add_argument(
    "--model_name_or_path",
    default="scutcyr/BianQue-1.0",
    type=str,
    required=True,
    help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
)
# 增加的special token
parser.add_argument(
    "--add_special_tokens",
    default=None,
    type=str,
    help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
)
# 训练的数据集csv格式文件，包含以下列：input、target、answer_choices、type
parser.add_argument(
    "--data_path",
    type=str,
    default="./data/cMedialog_example.csv",
    help='the path of the dataset for training model'
)
parser.add_argument(
    "--dataset_sample_frac", 
    default=1, 
    type=float, 
    help="数据集的采样率，范围：0~1"
)
parser.add_argument(
    "--train_radio_of_dataset", 
    default=0.94, 
    type=float, 
    help="数据集的训练样本比例，范围：0~1"
)
parser.add_argument(
    "--dataset_input_column_name", 
    default="input", 
    type=str, 
    help="column name of source text",
)
parser.add_argument(
    "--dataset_target_column_name", 
    default="target", 
    type=str, 
    help="column name of target text",
)

parser.add_argument(
    "--max_source_text_length",
    default=512,
    type=int,
    help="max length of source text, 512",
)
parser.add_argument(
    "--max_target_text_length",
    default=512,
    type=int,
    help="max length of target text, 512",
)

# 模型保存的路径
parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="The model checkpoint saving path",
)

parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="The seed setting.",
)
# 模型从output_dir继续运行
parser.add_argument(
    "--should_continue",
    action="store_true",
    help="Whether to continue from latest checkpoint in output_dir",
)
parser.add_argument(
    "--save_optimizer_and_scheduler",
    action="store_true",
    help="save optimizer and scheduler in the checkpoint",
)
parser.add_argument(
    "--overwrite_output_dir",
    action="store_true",
    help="Overwrite the content of the output directory",
)
parser.add_argument(
    "--no_cuda", action="store_true", help="Avoid using CUDA when available"
)

parser.add_argument(
    "--log_steps", default=10, type=int, help="logging output steps."
)


parser.add_argument(
    "--per_gpu_train_batch_size",
    default=1,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=1,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)

# 训练的优化器和学习率下降模式设置
parser.add_argument(
    "--optimizer",
    type=str,
    default="Adam",
    choices=['Adam', 'AdamW', 'Adafactor', 'Adafactor-srwf'],
    help="For optimizer.",
)
# 学习率下降模式
parser.add_argument(
    "--scheduler",
    type=str,
    default="get_constant_schedule",
    choices=['get_linear_schedule_with_warmup', 'get_constant_schedule_with_warmup', 'get_constant_schedule',
                'get_cosine_schedule_with_warmup', 'get_adafactor_schedule', 'no_schedule'],
    help="For scheduler.",
)
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--warmup_steps", default=8000, type=int, help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--warm_up_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps(warm_up_ratio*t_total)."
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)

# 模型训练的epoch数目
parser.add_argument(
    '--num_train_epochs',
    default=3,
    type=int,
)
# 模型训练的最大步数
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
# 设置是否进行迅雷
parser.add_argument(
    "--no_train", action="store_true", help="Only evaluate the checkpoint and not train"
)

# 模型的梯度加速
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
# Dataloder的num_workers
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help="num_workers for Dataloder",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="For distributed training: local_rank",
)
parser.add_argument(
    "--not_find_unused_parameters", action="store_true", help="If True set find_unused_parameters=False in DDP constructor"
)

parser.add_argument(
    "--save_total_limit",
    type=int,
    default=3,
    help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
)

# 混合精度训练
parser.add_argument("--autocast", action='store_true',
                    help="If true using autocast to automatically mix accuracy to accelerate training(开启自动混合精度加速训练)")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)