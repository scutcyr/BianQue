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

# File: run_train_model_bianque.sh
# Description: training model scripts
# Repository: https://github.com/scutcyr
# Mail: [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
# Date: 2023/03/15
# Usage:
# $ ./run_train_model_bianque.sh


# 路径配置
WORK_DIR="<The path of the file train_model.py>"
PRETRAINED_MODEL="scutcyr/BianQue-1.0"

# 指定csv格式数据集文件，其中csv文件当中input列为输入，target列为参考答案
PREPROCESS_DATA="$WORK_DIR/data/cMedialog_example.csv"

MODEL_TYPE=t5
MODEL_COMMENT=20230407_0600

# cd working path
cd $WORK_DIR

# 指定可以显卡，注意--nproc_per_node数目需要和这里的可用卡数一致
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 混合精度训练：--autocast
torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=9903 train_model.py \
    --model_type=t5 \
    --model_name_or_path=$PRETRAINED_MODEL \
    --data_path=$PREPROCESS_DATA \
    --dataset_sample_frac=1 \
    --train_radio_of_dataset=0.999 \
    --dataset_input_column_name=input \
    --dataset_target_column_name=target \
    --max_source_text_length=512 \
    --max_target_text_length=512 \
    --output_dir=$WORK_DIR/runs/${MODEL_TYPE}_${MODEL_COMMENT} \
    --seed=42 \
    --save_optimizer_and_scheduler \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --optimizer=AdamW \
    --scheduler=get_constant_schedule \
    --learning_rate=5e-5 \
    --num_train_epochs=1 \
    --save_total_limit=3 \
    --gradient_accumulation_steps=4 \
    --overwrite_output_dir \
    --not_find_unused_parameters


