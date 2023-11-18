# langchain_chatglm2-6b_Faiss

本次项目实验旨在针对chatglm2 设计webui，并构建知识库,项目主要参考了[thomas-yanxin/LangChain-ChatGLM-Webui: 基于LangChain和ChatGLM-6B等系列LLM的针对本地知识库的自动问答 (github.com)](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui)的设计，但是采用的最新的一些包，同时考虑不同的文档划分方式以及不同的知识库构建。

同时实现多文件的上传以及向量化的处理。

考虑 的知识库为：faiss(参考项目[1]中所采用的)，chromadb（参考项目[2]所采用的）和milvus(还没有实现)。



# 文件夹说明

1. 直接放在最外路径的几个文件，app.py等是针对的Faiss数据库
2. langchain_chromadb/milvus分别是针对chromadb和milvus的实验
3. 文件备份是对早期代码的一些备份
4. 笔记是总结的一些小知识
5. model_cache是从langchain-chatglm-webui这个项目中拷贝过来的
6. try是做的一些尝试，比较杂乱
7. vector_store 是存储向量数据库的地方
8. knowledge：里面是存放实验用的知识库文件包括：pdf,txt,docx,md文件


# detectron2，ntlk安装过程
## detectron2

首先 `git clone https://github.com/facebookresearch/detectron2.git`
然后 `pip install -e detectron2``
等待安装即可

在参考别人的安装方式 `pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2` 出现了几个问题
- linux上报错说找不到该文件
-  Windows上报错 'ERROR: Could not build wheels for detectron2, which is required to install pyproject.toml-based projects'

## ntlk

这个安装的时候首先需要安装：`pip intsall ntlk`

然后文件中有一个ntlk_data的文件，但我在ipynb中执行UnstructuredFileLoader,load一下的时候就报错

```
[nltk_data] Error loading punkt: [WinError 10060] 
[nltk_data]     由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。
```

应该是没有search到文件中的ntlk_data文件

后面我执行了try文件中的try.py文件，在我的c盘下面自动装了一个，然后就没有报错了

```python
import re
from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
loader = UnstructuredFileLoader(
    "try/1.pdf", strategy="fast", mode="elements"
)
docs = loader.load()
```

或者直接在代码中添加

```python
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")
                  ] + nltk.data.path
```

# 注意
1. gradio==3.50.2,最新的版本4.0.2有问题，报错： PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。不知道为什么会这样

2. gradio版本太低如gradio==3.23.0，也会报错：TypeError: __init__() got an unexpected keyword argument 'shape'

3. 服务器租用的是AutoDL的A5000，显存24G，内存30G

4. transformer==4.30.2，如果安装最新的版本（2023.11.2安装的时候版本是4/34.1）可能回有问题，有些库啥的没有
   1. AttributeError: 'ChatGLMTokenizer' object has no attribute 'tokenizer'
   2. 解决方案：降低版本如4.30.1（这也是chatglm2-6b github中 requirements中写的），或者更新tokenization_chatglm.py（暂时没有尝试过）：[AttributeError: 'ChatGLMTokenizer' object has no attribute 'tokenizer' · Issue #1835 · chatchat-space/Langchain-Chatchat (github.com)](https://github.com/chatchat-space/Langchain-Chatchat/issues/1835)

5. 文件中的：frpc_linux_amd64_v0.2，这个是用来为gradio创建外部链接的

   1. 对frpc_linux_amd64_v0.2进行执行能力赋予 chmod +x
      			1. 如果没有创建虚拟环境的话，路径一般是：/root/miniconda3/lib/python3.8/site-packages/gradio/（具体看miniconda的安装位置
   2. [【Gradio】Could not create share link-CSDN博客](https://blog.csdn.net/unp/article/details/131479915)

6. 要先加载知识，然后再向量化，最后才能问问问问题

   ```python
   #在app.py文件中 KnowledgeBasedChatLLM的get_knowledge_based_answer方法中
   def get_knowledge_based_answer(self,
                                  query,
                                  top_k: int = 6,
                                  history_len: int = 3,
                                  temperature: float = 0.01,
                                  top_p: float = 0.1,
                                  history=[]):
       self.llm.temperature = temperature
       self.llm.top_p = top_p
       self.history_len = history_len
       self.top_k = top_k
       prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
               如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。
   
               已知内容:
               {context}
   
               问题:
               {question}"""
       prompt = PromptTemplate(template=prompt_template,
                               input_variables=["context", "question"])
       self.llm.history = history[
           -self.history_len:] if self.history_len > 0 else []
       vector_store = FAISS.load_local(f'{VECTOR_STORE_PATH}/faiss_index', self.embeddings)
       #这里就意味着要先有文件的输入以及向量化：faiss_index,才能进行问答，不然就会报错
   
       knowledge_chain = RetrievalQA.from_llm(# 检索问答链
           llm=self.llm,
           retriever=vector_store.as_retriever(
               search_kwargs={"k": self.top_k}),
           prompt=prompt)
       knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
           input_variables=["page_content"], template="{page_content}")
   
       knowledge_chain.return_source_documents = True
   
       result = knowledge_chain({"query": query})
       return result
   # 如果没有按说明的顺序就会报错
   >>> RuntimeError: Error in faiss::FileIOReader::FileIOReader(const char*) at /project/faiss/faiss/impl/io.cpp:67: Error: 'f' failed: could not open vector_store/faiss_index/index.faiss for reading: No such file or directory
   
   ```

7. ~~注意在gradio中的input中最多只能有6个参数，因此这里没有考虑从chatglm的max_length参数~~

   1. ~~'\# /root/miniconda3/lib/python3.8/site-packages/gradio/utils.py:820: UserWarning: Expected maximum 6 arguments for function <function predict at 0x7f3b82df9c10>, received 7.'~~
   1. 这里是没有修改predict的参数，没有问题

8. 要实现多个文件的上传，需要修改gr.File中的参数 file_count='multiple'

   ```
   file = gr.File(label='请上传知识库文件',
                   file_types=['.txt', '.md', '.docx', '.pdf'],
                   file_count='multiple')
   ```

   1. 该参数有三个值*'single', 'multiple', 'directory'*，默认是single，表示上传一个文件；multiple表示上传多个文件；directory：表示上传一个文件夹目录，上传里面的所有文件
      1. 多个文件返回的是一个列表：[<tempfile._TemporaryFileWrapper object at 0x7f5ab3f395b0>, <tempfile._TemporaryFileWrapper object at 0x7f59e248a670>]，一个<>表示一个文件
      2. 单个文件直接返回的是 <tempfile._TemporaryFileWrapper object at 0x7f59e248a670>
         1. 对于单个文件可以通过.name 访问文件的地址，：/tmp/gradio/d7370fdd9ab92219d8e33f82c2fb68c39d7274fb/深度学习.txt，这是一个临时文件。但文件名和
            1. 具体可以参考gradio Files的type参数[Gradio File Docs](https://www.gradio.app/docs/file)
      3. 注意：如果用directory会警告：/root/miniconda3/lib/python3.8/site-packages/gradio/components/file.py:103: UserWarning: The\`file_types\` parameter is ignored when \`file_count` is 'directory'.
         1. 但是这里可以直接用，directory就是将文件夹中的文件全部传入，和multiple一样都是list

9. 如果要上传多个文件，还需要对向量数据库部分进行处理，如何处理多个向量，如何导入多个向量

   1. [langchain.vectorstores.faiss.FAISS — 🦜🔗 LangChain 0.0.329](https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html#langchain.vectorstores.faiss.FAISS.from_documents)
      1. *classmethod* **from_documents**(*documents: List[[Document](https://api.python.langchain.com/en/latest/schema/langchain.schema.document.Document.html#langchain.schema.document.Document)]*, *embedding: [Embeddings](https://api.python.langchain.com/en/latest/schema/langchain.schema.embeddings.Embeddings.html#langchain.schema.embeddings.Embeddings)*, ***kwargs: Any*) → VST（向量存储方式如IVF，PQ）
   
      2. 考虑到from_documents接受的是[document]
   
         1. 这里利用if+isinstance来判断传给init_knowledge_vector_store是否是列表
         2. 对于单个文件直接上传
         3. 多余多个文件，利用循环+列表的方法extend来将不同文件的document存在一个list中。
   
      3. ```python
         if isinstance(file_obj, list):
             docs=[]
             for file in file_obj:
                 doc=self.load_file(file.name)
                 docs.extend(doc)#这里不同于append，extend是将列表中的元素添加到另一个列表中
         else:
             docs = self.load_file(file_obj.name)
         ```
   
         



# 项目参考

[1] [thomas-yanxin/LangChain-ChatGLM-Webui: 基于LangChain和ChatGLM-6B等系列LLM的针对本地知识库的自动问答 (github.com)](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui)

[2] [IronSpiderMan/MachineLearningPractice: 机器学习实战案例，涉及机器学习、深度学习等各个方向。每个案例代码量在百行左右。 (github.com)](https://github.com/IronSpiderMan/MachineLearningPractice)

# 结果展示

模型加载

![image-20231113112751040](F:\大模型源码\实战\try1_practice1+2\img\Faiss_模型加载.png)

多文件加载，以及基于知识库的问答

![image-20231113113018587](F:\大模型源码\实战\try1_practice1+2\img\Faiss_问答.png)



# 向量数据库

这里的向量数据库仅局限于python 包的安装（也就是利用python进行操作），还没有向量数据库的部署，向量数据库的部署还得用docker，至于这个部署和维护还得继续学习
milvus是要docker的