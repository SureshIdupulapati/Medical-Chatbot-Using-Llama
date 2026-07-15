from flask import Flask, render_template, jsonify, request, Response
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from src.helper import *
from src.prompt import *
import os
import json
import queue
import threading

class QueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(("token", token))

    def on_llm_end(self, response, **kwargs) -> None:
        self.q.put(("end", None))

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.q.put(("error", str(error)))

app = Flask(__name__)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
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
    msg = request.form.get("msg") or request.args.get("msg")
    
    # Extract dynamic configuration
    try:
        temp = float(request.form.get("temperature") or request.args.get("temperature") or 0.8)
        max_tokens = int(request.form.get("max_new_tokens") or request.args.get("max_new_tokens") or 512)
        top_p = float(request.form.get("top_p") or request.args.get("top_p") or 0.95)
        k = int(request.form.get("k") or request.args.get("k") or 2)
        
        # Apply updates to LLM client config
        llm.client.config.temperature = temp
        llm.client.config.max_new_tokens = max_tokens
        llm.client.config.top_p = top_p
        
        # Apply updates to Retriever config
        retriever.search_kwargs['k'] = k
    except Exception as e:
        print(f"Error parsing configuration values: {e}")
        temp = 0.8
        max_tokens = 512
        top_p = 0.95
        k = 2
        
    input = msg
    print(f"Query: {input} (temp={llm.client.config.temperature}, max_tokens={llm.client.config.max_new_tokens}, k={retriever.search_kwargs.get('k')})")
    
    # 1. Simple Chit-chat Intent Classifier
    lowercase_msg = msg.strip().lower().rstrip("?").rstrip("!").rstrip(".") if msg else ""
    chitchat_greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy"}
    chitchat_identity = {
        "what is your name", "who are you", "what's your name", 
        "tell me your name", "what do they call you", "introduce yourself"
    }
    
    if lowercase_msg in chitchat_greetings:
        def gen_chitchat():
            msg1 = json.dumps({'type': 'source_documents', 'data': []})
            msg2 = json.dumps({'type': 'token', 'token': 'Hello! I am MediConsult AI, your clinical reference assistant. How can I help you today?'})
            yield f"data: {msg1}\n\n"
            yield f"data: {msg2}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        return Response(gen_chitchat(), mimetype='text/event-stream')
    elif lowercase_msg in chitchat_identity:
        def gen_identity():
            msg1 = json.dumps({'type': 'source_documents', 'data': []})
            msg2 = json.dumps({'type': 'token', 'token': 'I am MediConsult AI, a clinical reference assistant. I can help answer health-related and medical questions based on our clinical documentation.'})
            yield f"data: {msg1}\n\n"
            yield f"data: {msg2}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        return Response(gen_identity(), mimetype='text/event-stream')
        
    # 2. Similarity Search and Thresholding
    results = vectordb.similarity_search_with_score(input, k=k)
    
    # If no documents are retrieved or the best match is too distant:
    if not results or results[0][0].page_content == "" or results[0][1] > 1.1:
        print("Response: Query out-of-domain (similarity score below threshold).")
        def gen_fallback():
            msg1 = json.dumps({'type': 'source_documents', 'data': []})
            msg2 = json.dumps({'type': 'token', 'token': "I'm sorry, I couldn't find any relevant clinical documentation to answer that question. I can only assist with medical and health-related topics based on the reference book."})
            yield f"data: {msg1}\n\n"
            yield f"data: {msg2}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        return Response(gen_fallback(), mimetype='text/event-stream')
        
    # Standard document search and invocation via thread
    source_docs = []
    for doc, score in results:
        source_docs.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
        
    q = queue.Queue()
    handler = QueueCallbackHandler(q)
    
    def run_qa():
        try:
            qa.invoke({"query": input}, {"callbacks": [handler]})
        except Exception as e:
            q.put(("error", str(e)))
            
    threading.Thread(target=run_qa).start()
    
    def event_stream():
        # First send the source documents
        msg_docs = json.dumps({'type': 'source_documents', 'data': source_docs})
        yield f"data: {msg_docs}\n\n"
        
        # Then send the tokens
        while True:
            try:
                msg_type, val = q.get(timeout=30)
                if msg_type == "token":
                    msg_tok = json.dumps({'type': 'token', 'token': val})
                    yield f"data: {msg_tok}\n\n"
                elif msg_type == "end":
                    break
                elif msg_type == "error":
                    msg_err = json.dumps({'type': 'error', 'error': val})
                    yield f"data: {msg_err}\n\n"
                    break
            except queue.Empty:
                break
                
        yield "data: {\"type\": \"done\"}\n\n"
        
    return Response(event_stream(), mimetype='text/event-stream')
    

if __name__ == "__main__":
    app.run(debug=True)