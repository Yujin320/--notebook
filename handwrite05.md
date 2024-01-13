## 5. LMDeploy 的量化和部署

1. 安装lmdeploy：要指定安装[all]依赖包，默认安装的是[runtime]安装包
```sh
pip install 'lmdeploy[all]==v0.1.0'
```
2. 尝试安装包是否成功：import lmdeploy
3. 模型转换：想要使用TurboMind推理要先把模型转为TurboMind格式
   1. 在线转换：直接读取HuggingFace中的模型
```sh
lmdeploy chat turbomind internlm/internlm-chat-20b-4bit --model-name internlm-chat-20b
lmdeploy chat turbomind Qwen/Qwen-7B-Chat --model-name qwen-7b
#直接启动本地模型方法
lmdeploy chat turbomind /share/temp/model_repos/internlm-chat-7b/  --model-name internlm-chat-7b
#lmdeploy chat turbomind 本地路径 --model-name 名称
```
   2. 本地转换：将本地模型文件转换为LLMdeploy模型
```sh
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
#lmdeploy convert 模型名称 本地路径
#通过指定tp来确定生成的模型可以分发在几张显卡上
#在当前目录生层workspace文件夹，包含转换后文件
```
   3. TurboMind推理模型部署：
      1. 直接在命令行对话
```sh
lmdeploy chat turbomind ./workspace
```
      2. API服务
```sh
#启动API服务
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1
#访问API服务转发端口
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:23333
```
      3. 网页Demo演示
```sh
# Gradio+ApiServer
#必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True

#或直接和TurboMind连接
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```
      4. 直接通过python部署模型：
```python
from lmdeploy import turbomind as tm

# load model
model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-20b')
generator = tm_model.create_instance()
# process query
query = "你好啊兄嘚"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)
# inference
for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]

response = tm_model.tokenizer.decode(res.tolist())
print(response)
```
      5. 模型配置内容：./workspace/weights/config.ini
         1. quant_policy，默认值为 0，表示不使用 KV Cache，如果需要开启，则将该参数设置为 4。
         2. rope_scaling_factor，默认值为 0.0，表示不具备外推能力，设置为 1.0，可以开启 RoPE 的 Dynamic NTK 功能，支持长文本推理。另外，use_logn_attn 参数表示 Attention 缩放，默认值为 0，如果要开启，可以将其改为 1。
         3. 批处理大小：对应参数为 max_batch_size，默认为 64
1. 模型量化
   1. KVCache量化
      1. 计算minmax
```sh
# 计算 minmax
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_output
# 修改数据来源方式：
第一步：复制 calib_dataloader.py 到安装目录替换该文件：cp /root/share/temp/datasets/c4/calib_dataloader.py  /root/.conda/envs/lmdeploy/lib/python3.10/site-packages/lmdeploy/lite/utils/
第二步：将用到的数据集（c4）复制到下面的目录：cp -r /root/share/temp/datasets/c4/ /root/.cache/huggingface/datasets/
```
      2. 计算量化参数
```sh
# 通过 minmax 获取量化参数
lmdeploy lite kv_qparams \
  --work_dir ./quant_output  \
  --turbomind_dir workspace/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1
```
      3. 修改配置
         1. 修改 weights/config.ini 文件，这个我们在《2.6.2 模型配置实践》中已经提到过了（KV int8 开关），只需要把 quant_policy 改为 4 
         2. 如果用的是 TurboMind1.0，还需要修改参数 use_context_fmha，将其改为 0。
   1. W4A16量化
      1. 同3.3.1:计算minmax
```sh
# 计算 minmax
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_output
```
      2. 通过统计值拿到量化参数
```sh
# 量化权重模型
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output 
```
      1. 量化+转换为TurboMind格式
```sh
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128 \
    --dst_path ./workspace_quant
```
   1. 尝试流程：
Step1：优先尝试正常（非量化）版本，评估效果。
如果效果不行，需要尝试更大参数模型或者微调。
如果效果可以，跳到下一步。
Step2：尝试正常版本+KV Cache 量化，评估效果。
如果效果不行，回到上一步。
如果效果可以，跳到下一步。
Step3：尝试量化版本，评估效果。
如果效果不行，回到上一步。
如果效果可以，跳到下一步。
Step4：尝试量化版本+ KV Cache 量化，评估效果。
如果效果不行，回到上一步。
如果效果可以，使用方案。