# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
# from langchain.chains import ChatVectorDBchain 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
                
    return file_list

# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)

    print(f'file_lst_length = {len(file_lst)}')
    print('---------------------------------------------------------------------------------')
    print(f'file_lst = {file_lst}')
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'pdf':
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs

# 目标文件夹
# tar_dir = [
#     "/root/data/Agriculture_KnowledgeGraph",
#     "/root/data/Agriculture-KBQA",
#     "/root/data/KnowledgeGraph_Agriculture",
#     "/root/data/KG-Crop",
#     "/root/data/wheat-knowledge",
# ]

# tar_dir = [
#     "/root/code/paper_demo/2023年05期"
# ]

tar_dir = [
    '/root/code/paper_demo/01.小麦',
    '/root/code/paper_demo/02.水稻',
    '/root/code/paper_demo/03.玉米',
    '/root/code/paper_demo/04.棉花',
    '/root/code/paper_demo/a.植物保护学报',
    '/root/code/paper_demo/b.植物病理学报'
    ]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))
print(f'docs = {docs}')

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
print(f'text_splitter = {text_splitter}')
split_docs = text_splitter.split_documents(docs)
print(f'split_docs = {split_docs}')

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
# print(f'embeddings = {embeddings}')
# 构建向量数据库
# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'
print(f'persist_directory = {persist_directory}')
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
print(f'vectordb = {vectordb}')
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()