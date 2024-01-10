# Homework03
## -- 基于 InternLM 和 LangChain 搭建你的知识库

1. 下载模型过程

2. 运行截图


## -- 基于自己的知识内容搭建问答机器人

1. 数据导入和数据库构建代码
```python
from tqdm import tqdm
from langchain.document_loaders.json_loader import JSONLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

data_dir=''
def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".json"):
                file_list.append(os.path.join(filepath, filename))
                print(filename)
    return file_list

def get_text(dir_path):
    file_lst = get_files(dir_path)
    docs = []
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == '.json':
            loader=JSONLoader(one_file)
        else:
            continue
        docs.extend(loader.load())
    return docs

docs=get_text(data_dir)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")


persist_directory = 'data_base/vector_db/ECNU_chroma'
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectordb.persist()
```
2. 问答链构造代码
```python
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
persist_directory = 'data_base/vector_db/ECNUchroma'
vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)

template = """使用以下上下文来回答用户关于华东师范大学问题。如果你不知道答案，就说你不知道。总是使用提问语言回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答你不知道，如果用户提问与华东师范大学无关请减少对上下文的依赖尝试直接回答。
有用的回答:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
```
