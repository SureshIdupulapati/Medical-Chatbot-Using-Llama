from flask import Flask, render_template, jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.helper import *
from src.prompt import *
import os

app = Flask(__name__)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model=r"C:\Users\Sures\jupyter_programs\Open-AI\Medical-Chatbot-Using-Llama2\Model\llama-2-7b.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                        'temperature':0.8}
                )

persist_directory = "db"
embeddings = download_hugging_face_embeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={'k': 2})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query":input})
    print("Response: ",result["result"])
    return str(result["result"])
    

if __name__ == "__main__":
    app.run(debug=True)