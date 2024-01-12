## 4. xturner微调工具箱--微调大模型

1. 环境配置：源码安装xturner：
```sh
mkdir xtuner019 && cd xtuner019
git clone -b v0.1.9  https://github.com/InternLM/xtuner
cd xtuner
pip install -e '.[all]'

#复现工作的尝试文件夹
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```
2. 微调准备：
   1. 配置文件查看
```sh
xtuner list-cfg
```
   2. 拷贝课程中使用的配置文件到工作文件夹
    internlm_chat_7b_qlora_oasst1_e3:模型名	internlm_chat_7b
                                    使用算法	qlora
                                    数据集	oasst1
                                    把数据集跑几次	跑3次：e3 (epoch 3)
```sh
#xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
cd ~/ft-oasst1
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```
   1. 准备模型文件（将已经下载好的模型文件夹软连接到工作文件夹）
```sh
ln -s /bin/less /usr/local/bin/less
```
   2. 数据集下载：从share文件夹中直接复制
```sh
cd ~/ft-oasst1
cp -r /root/share/temp/datasets/openassistant-guanaco .
```
   3. 修改模版的配置文件内容：模型和数据集路径改为本机中的路径
```sh
# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'
# 修改训练数据集为本地路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = './openassistant-guanaco'
```
1. 开始微调训练：
   1. 微调开始：得到微调结果的权重文件
```sh
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py
```
   2. 微调权重文件转换为HF模型：LoRA模型文件
```sh
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```
   3. 将LoRA模型文件合并到InternLM
```sh
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```
   4. 尝试对话
```sh
xtuner chat ./merged --prompt-template internlm_chat
# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
- model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"
+ model_name_or_path = "merged"
python ./cli_demo.py
```
1. 自定义数据集微调
   1. 数据转换为XTurner数据格式.jsonL
```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```
   2. 划分训练集和测试集
```python

```
   3. 自定义微调
      1. 自定义数据集：
```sh
cp ~/tutorial/xtuner/MedQA2019-structured-train.jsonl .
```
      2. 准备配置文件
```sh
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_medqa2019_e3.py
```
      3. 修改配置文件
```sh
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'MedQA2019-structured-train.jsonl'

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
```