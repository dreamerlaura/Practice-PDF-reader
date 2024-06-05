from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def qa_agent(openai_api_key,memory,uploaded_file,question):
    model=ChatOpenAI(model='gpt-3.5-turbo',openai_api_key=openai_api_key,openai_api_base = "https://api.aigc369.com/v1")

    # 文件路径传给文件加载器
    file_content= uploaded_file.read() #对用户传入的文档内容进行读取，会返回bytes
    temp_file_path='temp.pdf' #新建一个临时文件路径
    with open(temp_file_path,'wb') as temp_file:  #文件写入 wb表示写入二进制，后面表示文件的变量名自定
        temp_file.write(file_content)  #调用write进行写入，路径对应的文件就存有用户上传的PDF内容
    loader=PyPDFLoader(temp_file_path) #加载器实例
    docs=loader.load() #加载出的documents列表

    #分割
    text_splitter=RecursiveCharacterTextSplitter(   #实例化分割器
        chunk_size=1000,
        chunk_overlap=50,
        separators=['\n','。','！','？','，','、','']
    )
    texts=text_splitter.split_documents(docs)  #调用方法

    #向量嵌入,这里与教程有所不同，需要传入密钥和基地
    embeddings_model=OpenAIEmbeddings(openai_api_key=openai_api_key,
                                      openai_api_base = "https://api.aigc369.com/v1")  #实例化嵌入模型

    #向量数据库
    db=FAISS.from_documents(texts,embeddings_model)  #调用方法，传入前面的分割文档和嵌入模型

    #检索器
    retriever=db.as_retriever()

    #创建带记忆的链
    qa=ConversationalRetrievalChain.from_llm(   #调用链的方法
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response=qa.invoke({'chat_history':memory,'question':question})  #传入字典

    #返回链的调用结果
    return response