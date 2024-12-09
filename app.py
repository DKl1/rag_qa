import os

import gradio as gr
from datasets import load_dataset

from rag import QuestionAnsweringBot

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to RAG!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""


def read_docs():
    ds = load_dataset("rajpurkar/squad")
    context_data = list(set(ds['train']['context'][:100]))

    return context_data


def answer_question(question_, search_method_, open_ai_key_):
    os.environ["OPENAI_API_KEY"] = open_ai_key_
    docs = read_docs()
    bot = QuestionAnsweringBot(docs, search_method_, )
    answer_question_ = bot.answer_question(question_)
    return answer_question_, bot.retriever_documents, bot.reranker_documents


with gr.Blocks(js=js) as demo:
    open_ai_key = gr.Text(label="OpenAI Key", placeholder="Enter your OpenAI API key here")
    search_method = gr.Radio(
        choices=["Keyword Search (BM25)", "Semantic Search", "Hybrid (BM25 + Semantic)"],
        label="Retriever Method",
        value="Hybrid (BM25 + Semantic)",  # Default selection
    )
    question = gr.Text(label="Question", placeholder="Ask your question here")
    answer = gr.Text(label="Model Output", interactive=False, lines=5)

    retriever_documents = gr.Textbox(label="Retriever Documents", interactive=False, lines=10)
    reranker_documents = gr.Textbox(label="Reranker Documents", interactive=False, lines=10)


    def format_outputs(documents):
        if isinstance(documents, list):
            return "\n".join([f"- {doc}" for doc in documents])
        return documents


    def formatted_answer_question(question_, search_method_, open_ai_key_):
        ans, retriever_docs, reranker_docs = answer_question(question_, search_method_, open_ai_key_)
        retriever_docs_formatted = format_outputs(retriever_docs)
        reranker_docs_formatted = format_outputs(reranker_docs)
        return ans, retriever_docs_formatted, reranker_docs_formatted


    button_rag = gr.Button('Get answer')
    button_rag.click(formatted_answer_question, inputs=[question, search_method, open_ai_key],
                     outputs=[answer, retriever_documents, reranker_documents])

demo.launch()
