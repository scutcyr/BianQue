# coding=utf-8
# Copyright 2023 South China University of Technology and 
# Engineering Research Ceter of Ministry of Education on Human Body Perception.
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


# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2023.03.14

''' 运行方式
```bash
pip install streamlit # 第一次运行需要安装streamlit
pip install streamlit_chat # 第一次运行需要安装streamlit_chat
streamlit run bianque_v1_app.py --server.port 9001
```

## 测试访问

http://<your_ip>:9001

'''


import os
import torch
import streamlit as st
from streamlit_chat import message
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_or_path = "scutcyr/BianQue-1.0"
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(user_history, bot_history, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样
    max_new_tokens=512 lost...'''

    if len(bot_history)>0:
        context = "\n".join([f"病人：{user_history[i]}\n医生：{bot_history[i]}" for i in range(len(bot_history))])
        input_text = context + "\n病人：" + user_history[-1] + "\n医生："
    else:
        input_text = "病人：" + user_history[-1] + "\n医生："
        return "我是利用人工智能技术，结合大数据训练得到的智能医疗问答模型扁鹊，你可以向我提问。"
    
    input_text = preprocess(input_text)
    print(input_text)
    encoding = tokenizer(text=input_text, truncation=True, padding=True, max_length=768, return_tensors="pt").to(device) 
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    print('医生: '+postprocess(out_text[0]))
    return postprocess(out_text[0])

st.set_page_config(
    page_title="扁鹊健康支持模型（BianQue-1.0）",
    page_icon="⏺️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """     
-   版本：扁鹊健康大模型（BianQue） V1.0.0 Beta
-   机构：广东省数字孪生人重点实验室
-   作者：陈艺荣、王振宇、徐志沛、方凱、李思航、王骏宏、邢晓芬、徐向民
	    """
    }
)

st.header("⏺️扁鹊健康支持模型（BianQue-1.0） ")

with st.expander("ℹ️ - 关于我们", expanded=False):
    st.write(
        """     
-   版本：扁鹊健康大模型（BianQue） V1.0.0 Beta
-   机构：广东省数字孪生人重点实验室
-   作者：陈艺荣、王振宇、徐志沛、方凱、李思航、王骏宏、邢晓芬、徐向民
	    """
    )

# https://docs.streamlit.io/library/api-reference/performance/st.cache_resource
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model.to(device)
    print('Model Load done!')
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    print('Tokenizer Load done!')
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


user_col, ensure_col = st.columns([5, 1])

def get_text():
    input_text = user_col.text_area("请在下列文本框输入您的咨询内容：","", key="input", placeholder="请输入您的咨询内容，并且点击Ctrl+Enter(或者发送按钮)确认内容")
    if ensure_col.button("发送", use_container_width=True):
        if input_text:
            return input_text  

user_input = get_text()

if user_input:
    st.session_state.past.append(user_input)
    output = answer(st.session_state['past'],st.session_state["generated"])
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        if i == 0:
            # 
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            message(st.session_state["generated"][i]+"\n\n------------------\n以下回答由扁鹊健康模型自动生成，仅供参考！", key=str(i), avatar_style="avataaars", seed=5)
        else:
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            #message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["generated"][i], key=str(i), avatar_style="avataaars", seed=5)

if st.button("清理对话缓存"):
    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
    st.session_state['generated'] = []
    st.session_state['past'] = []