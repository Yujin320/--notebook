## 6 OpenCompass:大模型评测

1. 测评对象包括语言大模型和多模态大模型
   1. 语言大模型包括基座模型和对话模型
2. 测试工具架构
   1. 主要面向基座模型和对话模型
   2. 测评能力：语言安全等通用模型能力+代码，长文本，工具调用特色能力
   3. 方法：
      1. 客观评测：可以通过选择调控方式批量测评并得到测评结果
      2. 主观测评：评估用户对模型回复的满意度
   4. 工具：分布式测评，提示工程，评测报告和榜单
3. 评测方法：
   1. 客观评测：具有标准答案的客观问题，通过计算模型输出与标准答案差异来衡量模型性能。输入和输出进行了一定规范设计，防止自由度带来噪音
      1. 判别式：大模型对评测给出的多个答案给出困惑度，评测选出困惑度最小的一个
      2. 生成式：模型响应输出后进行进一步处理以满足数据集要求
   2. 主观评测：
      1. 对安全性和特殊能力效果的评测
      2. 单模型回复满意度统计和基于模型打分的主观评测
4. 快速开始：
   1. 配置：
      1. 选择模型和数据集
      2. 选择评估策略，计算后端
      3. 定义结果显示方式
   2. 推理和评估：
      1. 让模型从数据集产生输出
      2. 衡量输出与标准答案的匹配程度
   3. 可视化：8️把结果当成可用的表格，将其保存为csv文件等
5. 实践一个InternLM在C-Eval基准任务上的评估：
   1. 安装opencampass
```sh
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```
   2. 数据准备
```sh
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```
   3. 查看支持的数据集和模型(试试有没有安装成功)
```sh
python tools/list_configs.py internlm ceval
```
   4. 启动评测
```sh
--datasets ceval_gen \
--hf-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace 模型路径
--tokenizer-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 2048 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 4  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug
```