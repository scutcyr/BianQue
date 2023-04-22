# [扁鹊(BianQue)]((https://github.com/scutcyr/BianQie))

## 最近更新
* [扁鹊-1.0：通过混合指令和多轮医生问询数据集的微调，提高医疗聊天模型的“问”能力（BianQue-1.0: Improving the "Question" Ability of Medical Chat Model through finetuning with Hybrid Instructions and Multi-turn Doctor QA Datasets）](https://huggingface.co/scutcyr/BianQue-1.0)


## 简介

**扁鹊-1.0（BianQue-1.0）**是一个经过指令与多轮问询对话联合微调的医疗对话大模型。我们经过调研发现，在医疗领域，往往医生需要通过多轮问询才能进行决策，这并不是单纯的“指令-回复”模式。用户在咨询医生时，往往不会在最初就把完整的情况告知医生，因此医生需要不断进行询问，最后才能进行诊断并给出合理的建议。基于此，我们构建了**扁鹊-1.0（BianQue-1.0）**，拟在**强化AI系统的问询能力**，从而达到模拟医生问诊的过程。我们把这种能力定义为“望闻问切”当中的“问”。

综合考虑当前中文语言模型架构、参数量以及所需要的算力，我们采用了[ClueAI/ChatYuan-large-v2](https://huggingface.co/ClueAI/ChatYuan-large-v2)作为基准模型，在8张 NVIDIA RTX 4090显卡上微调了1个epoch得到**扁鹊-1.0（BianQue-1.0）**，用于训练的**中文医疗问答指令与多轮问询对话混合数据集**包含了超过900万条样本，这花费了大约16天的时间完成一个epoch的训练。

我们将计划围绕扁鹊模型的“望闻问切”能力，结合医学专家知识、多模态技术、多生理信号计算等，进行多个版本的模型迭代研究。

扁鹊（BianQue）模型欢迎你的贡献！我们鼓励你在 [BianQue GitHub](https://github.com/scutcyr/BianQue) 页面报告问题、贡献 PR 并参与讨论。我们期待与更多的高校、医院、研究实验室、公司等进行合作，共同开展下一代扁鹊模型研究。对于此类需求（以及其他不适合在 GitHub 上提出的需求），请直接发送电子邮件至 [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)。



## 训练数据
我们结合当前开源的中文医疗问答数据集（[MedDialog-CN](https://github.com/UCSD-AI4H/Medical-Dialogue-System)、[IMCS-V2](https://github.com/lemuria-wchen/imcs21)、[CHIP-MDCFNPC](https://tianchi.aliyun.com/dataset/95414)、[MedDG](https://tianchi.aliyun.com/dataset/95414)、[cMedQA2](https://github.com/zhangsheng93/cMedQA2)、[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)），以及自建的指令数据集，通过进一步的数据清洗，构建了一个大于900万条样本的**中文医疗问答指令与多轮问询对话混合数据集**，数据集的平均轮数为3，最大轮数达到218，数据格式为：
```data
input: "病人：六岁宝宝拉大便都是一个礼拜或者10天才一次正常吗，要去医院检查什么项目\n医生：您好\n病人：六岁宝宝拉大便都是一个礼拜或者10天才一次正常吗，要去医院检查什么项目\n医生：宝宝之前大便什么样呢？多久一次呢\n病人：一般都是一个礼拜，最近这几个月都是10多天\n医生：大便干吗？\n病人：每次10多天拉的很多\n医生："
target: "成形还是不成形呢？孩子吃饭怎么样呢？"
```
训练数据当中混合了大量target文本为**医生问询的内容**而非直接的建议，这将有助于提升AI模型的问询能力。




## 模型“问”能力示例
“望闻问切”四诊法由扁鹊发明。“四诊法”（望、闻、问、切），是中医诊病的基本方法。其中，“问”的能力可以被建模到语言模型当中。我们把AI模型的“问”能力定义为：    
**“问”能力**：AI模型能够根据病人输入的信息，结合医学知识来进行**恰当的问询**，并且结合进一步的问询得到的病人的信息，给出“合适的诊断或建议”。“问”的过程通常是一个多轮对话过程。


以下是一个简单的例子：

* **例子1**    
用户：我最近感觉全身疲惫   
扁鹊-1.0：是劳累了，还是熬夜了？    
用户：这周都在熬夜赶论文    
扁鹊-1.0：嗯，熬夜容易引起各种不适，如头晕、乏力、失眠、多梦等，多休息，可以口服谷维素、维生素B1和安神补脑液，避免熬夜。    

* **例子2**   
用户：我最近经常失眠    
扁鹊-1.0：失眠多久了？    
用户：两周了    
扁鹊-1.0：睡眠怎么样？    
用户：上床几小时才睡得着    
扁鹊-1.0：可以口服一些药物   
用户：有具体推荐吗   
扁鹊-1.0：需要先检查一下   
用户：那我需要做什么检查？   
扁鹊-1.0：心电图   


## 使用方法



```python
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = T5Tokenizer.from_pretrained("scutcyr/BianQue-1.0")
model = T5ForConditionalGeneration.from_pretrained("scutcyr/BianQue-1.0")


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

answer_text = answer(user_history=["你好！",
                                   "我最近经常失眠",
                                   "两周了",
                                   "上床几小时才睡得着"], 
                     bot_history=["我是利用人工智能技术，结合大数据训练得到的智能医疗问答模型扁鹊，你可以向我提问。",
                                  "失眠多久了？",
                                  "睡眠怎么样？"])
```

## 声明

**扁鹊-1.0（BianQue-1.0）**当前仅经过1个epoch的训练，尽管模型具备了一定的医疗问询能力，但其仍然存在以下局限：
* 训练数据来源于开源数据集以及互联网，尽管我们采用了严格的数据清洗流程，数据集当中仍然不可避免地存在大量噪声，这会使得部分回复产生错误；
* 医生“问询”是一项复杂的能力，这是非医生群体所不具备的，当前的模型对于模拟“医生问询”过程是通过大量样本学习得到的，因此在问询过程当中，有可能出现一些奇异的提问风格。换一句话来说，当前版本的模型强化了“问”的能力，但是“望”、“闻”、“切”的能力仍待进一步研究！


## 引用
```bib
@article{chen2023bianque1,
      title={BianQue-1.0: Improving the "Question" Ability of Medical Chat Model through finetuning with Hybrid Instructions and Multi-turn Doctor QA Datasets}, 
      author={Yirong Chen and Zhenyu Wang and Xiaofen Xing and Zhipei Xu and Kai Fang and Sihang Li and Junhong Wang and Xiangmin Xu},
      year={2023},
      url={https://github.com/scutcyr/BianQue}
}
```
