import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.title('AI智能PDF文档工具')
with st.sidebar:
    password=st.text_input('请输入API密钥： ',type='password')

if 'memory' not in st.session_state:
    st.session_state['memory']=ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',  #对应链中memory的键
        output_key='answer'  #对应输出结果的键
    )

uploaded_file=st.file_uploader('上传你的PDF文件',type='pdf')
question=st.text_input('对PDF进行提问',disabled=not uploaded_file)

if uploaded_file and question and not password:
    st.info('请输入你的API密钥')
    st.stop()

if uploaded_file and question and password:
    with st.spinner('AI正在思考中，请稍后...'):
        response=qa_agent(password,st.session_state['memory'],uploaded_file, question)

    st.write('### 答案')
    st.write(response['answer'])
    st.session_state['chat_history']=response['chat_history']

if 'chat_history' in st.session_state:
    with st.expander('历史消息'):
        for i in range(0,len(st.session_state['chat_history']),2):
            human_message=st.session_state['chat_history'][i]
            ai_message=st.session_state['chat_history'][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state['chat_history'])-2:
                st.divider()