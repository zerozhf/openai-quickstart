import gradio as gr
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(
        api_key="sk-fYThF6OhEdD6QgFyCeB0C820728147439283Bc5529BeFd68",
        base_url="https://api.xiaoai.plus/v1"
    ),
                          allow_dangerous_deserialization=True)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key="sk-fYThF6OhEdD6QgFyCeB0C820728147439283Bc5529BeFd68",
        base_url="https://api.xiaoai.plus/v1"
    )
    template = """你是一个顶级的手机销售，你需要回答用户关于购买手机上的问题，只回答与购买手机相关的问题，不能暴露你是机器人.
        {context}
        Question: {question}
        Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            chain_type_kwargs={"prompt": prompt},
                                            return_source_documents=True,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "GG"

def ask_question(questtion):
    print(f"[questtion]{questtion}")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key="sk-fYThF6OhEdD6QgFyCeB0C820728147439283Bc5529BeFd68",
        base_url="https://api.xiaoai.plus/v1"
    )

    prompt = PromptTemplate(
        input_variables=["question"],
        template="给回答这个问题：{question}",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.invoke({
        'question': questtion
    }))
    return chain.invoke({
        'question': questtion
    })




    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="手机销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    initialize_sales_bot()
    launch_gradio()
