import gradio as gr
from haystack.nodes import TransformersReader, PromptNode
from haystack.pipelines import Pipeline
from transformers import GenerationConfig

# custom imports
from main import document_store_import, config_import_yaml, get_prompt
from doc.huggingface_webscraper import get_model_names

config = config_import_yaml() # import config
document_store, documents, labels = document_store_import(config, 'demo')
prompt_lfqa = get_prompt('lfqa')

from haystack.nodes import BM25Retriever
retriever = BM25Retriever(document_store=document_store)

# reader = TransformersReader(config['reader']['name'])
# prompt = get_prompt('question-answering')
# generator = PromptNode(default_prompt_template='question-answering')
#model_names = get_model_names('text2text-generation')
model_name_or_path = config['generator']['name']
GENERATION_KWARGS = {
    "generation_kwargs": GenerationConfig(
        do_sample=config['generator']['do_sample'], 
        top_p=config['generator']['top_p'], 
        temperature=config['generator']['temperature'],
        #max_length=config['generator']['max_length'],
        max_new_tokens=config['generator']['max_new_tokens'],
        early_stopping=config['generator']['early_stopping'],
    ),
    "max_tokens": config['generator']['max_tokens'],
}
generator = PromptNode(default_prompt_template=prompt_lfqa, model_name_or_path=model_name_or_path, model_kwargs=GENERATION_KWARGS)

pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
#pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])
pipe.add_node(component=generator, name="Generator", inputs=["Retriever"])

def chat_bot(inp="What is kubernetes?", **kwargs):
    result = pipe.run(inp, **kwargs)
    return result['answers'][0].answer



chat_bot("What is kubernetes? How can i use it to deploy my application?")

# Create the Gradio interface
title = "Chat Bot Open Domain Question Answering (ODQA)"
description = f"""In this thesis, we aim to compare the performance of State-Of-The-Art Language Models
in a Zero-Shot Open Domain Question Answering setting for technical topics, specifically
regarding cloud technology and containerization. Question Answering has historically
been mostly extractive in nature, but in recent years we have seen the paradigm of Natural Language Processing switch towards the more abstract Natural Language Generation
approach. We propose a two-step architecture, in which the solution attempts to answers
questions from a set of documents with no prior training or fine-tuning. We do not solely
focus on Retriever-Reader methods (e.g., BERT, RoBERTa), but also evaluate RetrieverGenerator (e.g., GPT, FLAN-T5) systems through Long Form Question Answering. The
Amazon Web Services dataset is used as an benchmark for evaluating performance of the
zero-shot Open Book Question Answering system [1]. Empirical results are sometimes
obtained by splitting the documents in to smaller subsections like paragraphs or passages, therefore we analyse the hyperparameters for document splitting using a sliding
window. We show that RoBERTa-large is able to achieve a new State-Of-The-Art F1 score
of 59.19 through proper pre-processing of the documents and carefully selecting hyperparameters, gaining a respectful 18.66 compared to the baseline and 16.99 compared to
the best results in the original study. We conclude that generative models and Long Form
Question Answering demonstrate great potential, but come with their own set of biases
and risks. We observe that when the complexity of the model far exceeds the evaluation
metrics, the relevance and meaning of the metrics become questionable. In this context,
Semantic Answer Similarity and METEOR prove useful for analyzing diverse model outputs, as they are not dependent on lexical stride like ROUGE, BLEU, F1 and EM. Splitting
documents into passages offers performance benefits, although it is important to note that
document splitting may not necessarily be superior for all use cases and the optimal hyper parameter values are expected to vary depending on the specific application.






`What is the command to see the logs of a pod?`
`Is Amazon EBS encryption available on M3 instances?`


"""

input_text = gr.inputs.Textbox(label=f"Enter a question and get an answer from {model_name_or_path}. ")
output_text = gr.outputs.Textbox(label="Answer")
#smodel_dropdown = gr.inputs.Dropdown(model_names, label="Model")

demo = gr.Interface(fn=chat_bot, inputs=input_text, outputs=output_text, title=title, description=description)
demo.launch()