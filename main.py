import logging
import os
import json
import gc
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Machine Learning
from transformers import GenerationConfig
from evaluate import load, EvaluationModule
from sentence_transformers import CrossEncoder
from haystack.pipelines import Pipeline, DocumentSearchPipeline
from haystack.schema import Document, Label, MultiLabel, Answer, EvaluationResult
from haystack.utils import print_answers, print_documents, print_questions
from haystack.nodes import BaseRetriever, BM25Retriever, TfidfRetriever, EmbeddingRetriever, DensePassageRetriever
from haystack.nodes import BaseReader, BaseGenerator, FARMReader, TransformersReader, PromptNode, PromptTemplate, AnswerParser
from haystack.nodes.prompt.prompt_template import get_predefined_prompt_templates

# Custom modules
from utils import stackexchange_documents, aws_docs_import_qa, aws_docs_documents, aws_docs_labels, kubernetes_documents, hash_md5, normalize_answer, f1_score, get_key

logger = logging.getLogger("ads-llm-qa")

OPENAI_MODEL_NAMES = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4']

# Functions
def document_import(sources:list=['kubernetes', 'stackexchange', 'aws'], 
                    modalities:list[str]=['text', 'table']) -> list[Document]:
    "Import documents from various sources, e.g. kubernetes, stackexchange, aws"
    if isinstance(sources, str):
        sources = [sources]
    elif not isinstance(sources, list):
        raise TypeError(f"sources should be a list of strings, not {type(sources)}")
    documents = []
    logger.info(f"Importing documents...")
    if 'kubernetes' in sources:
        logger.info(f"Importing kubernetes documents...")
        kubernetes_docs = kubernetes_documents(file_path='./data/kubernetes_docs.json')
        documents += kubernetes_docs
        kubernetes_blog = kubernetes_documents(file_path='./data/kubernetes_blog.json')
        documents += kubernetes_blog
    if 'stackexchange' in sources:
        logger.info(f"Importing stackexchange documents...")
        stackexchange = stackexchange_documents(db='./data/stackexchange_kubernetes.db')    
        documents+=stackexchange
    if 'aws' in sources:
        logger.info(f"Importing aws documents...")
        aws_docs = aws_docs_documents(base_dir='./data/aws-documentation/', modalities=modalities)
        documents += aws_docs
    return documents

def config_import_yaml(file_path:str=f"main.yml") -> dict:
    "Import runtime configuration from a `.yml` file."
    logger.info(f"Loading configuration from `{file_path}`...")
    with open(file_path, "r") as stream:
        config = yaml.safe_load(stream) 
    logger.info(f"Configuration loaded.")
    return config 

def config_hash(config:dict) -> str:
    "Hash the runtime settings to avoid repeating same analysis. "
    h = hash_md5(
        str(config['data']['data_sources']) + str(config['data']['data_modalities']) + \
        str(config['document_store']['embedding_dim']) + str(config['pre_processing']['split']['size'])+ \
        str(config['pre_processing']['split']['stride']) + str(config['pre_processing']['split']['respect_boundary'])
    )
    h = 'baseline' if config['pre_processing']['split']['by'] is None else h
    return h

def document_store_import(config:dict, index_name:str):
    "Import document store (database/back-end) based on configuration file and index name for the documents."
    logger.info(f"Attempting to connect to {config['document_store']['backend']} backend...")
    if config['document_store']['backend']=='memory':
        from haystack.document_stores import InMemoryDocumentStore
        document_store = InMemoryDocumentStore(use_bm25=True)   
        logger.warning('Using InMemoryDocumentStore, progress will not be saved...')
    elif config['document_store']['backend']=='faiss':
        from haystack.document_stores import FAISSDocumentStore
        from haystack.utils import print_documents
        faiss_index = config['document_store']['faiss']['index_path']
        faiss_config = config['document_store']['faiss']['config_path']
        if os.path.exists(faiss_index) and os.path.exists(faiss_config):
            document_store = FAISSDocumentStore.load(
                index_path=faiss_index, 
                config_path=faiss_config)
        else:
            document_store = FAISSDocumentStore(
                embedding_dim=config['document_store']['embedding_dim'], 
                faiss_index_factory_str=config['document_store']['faiss']['index_factory_str'], 
                return_embedding=True, 
                sql_url=config['document_store']['faiss']['database_path'], 
                index=f'document_{index_name}',
                similarity=config['document_store']['similarity'])
    elif config['document_store']['backend']=='elasticsearch':
        # Start Elasticsearch using Docker in the background if it's not running yet
        if config['document_store']['elasticsearch']['host']=='localhost':
            from haystack.utils import launch_es
            launch_es() 
        from haystack.document_stores import ElasticsearchDocumentStore
        document_store = ElasticsearchDocumentStore(
            host=config['document_store']['elasticsearch']['host'],
            port=config['document_store']['elasticsearch']['port'],
            scheme=config['document_store']['elasticsearch']['scheme'],
            ca_certs=config['document_store']['elasticsearch']['certificate_path'], 
            username=config['document_store']['elasticsearch']['username'],
            embedding_dim=config['document_store']['embedding_dim'],    
            password=config['document_store']['elasticsearch']['password'], 
            index = f"document_{index_name}")  
    else:
        raise ValueError(f"Backend {config['document_store']['backend']} not supported.")  
    logger.info(f"Connected to {config['document_store']['backend']} backend...")
    # Try to import the documents and labels. If none are found import from source.
    logger.info("Importing documents from document store...")
    documents, labels = document_store.get_all_documents(), document_store.get_all_labels()
    embedding_count = document_store.get_embedding_count()
    if len(documents)==0:
        logger.warning("Document store is empty. Importing documents from source...")
        # Import documents from source
        #document_store.delete_documents()
        documents_raw = document_import(
            sources=config['data']['data_sources'], 
            modalities=config['data']['data_modalities'])
        #document_store.write_documents(documents_raw)
        # Import labels from source
        if len(labels)==0:
            labels = aws_docs_labels(documents_raw)
            document_store.write_labels(labels)
        # Preprocess documents
        from haystack.nodes import PreProcessor
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by=config['pre_processing']['split']['by'],
            split_length=config['pre_processing']['split']['size'],
            split_overlap=config['pre_processing']['split']['stride'],
            split_respect_sentence_boundary=config['pre_processing']['split']['respect_boundary'])
        documents = preprocessor.process(documents_raw)
        document_store.write_documents(documents, duplicate_documents='overwrite')  
    logger.info(f"Document store contains {len(documents)} documents and {len(labels)} labels. \
                {round(embedding_count/len(documents)*100, 2)}% of the documents have embeddings ({embedding_count}).")
    return document_store, documents, labels

def get_prompt(template_name:str):
    """Get a haystack prompt template by name. 
    Use default prompt templates first, then use custom prompt templates.
    Prompt Template Long-Form Question Answering (LFQA) was slightly edited (question is before the documents to avoid truncating it).
    Source: https://haystack.deepset.ai/tutorials/12_lfqa"""
    default_prompts = get_predefined_prompt_templates()
    print("Available prompt templates:")
    for p in default_prompts:
        print('\t- ' + p.name) 
    if template_name in [p.name for p in default_prompts]:
        prompt = [p for p in default_prompts if p.name==template_name][0]
    elif template_name=='lfqa':
        prompt = PromptTemplate(name="lfqa",
                                prompt_text="""Synthesize a comprehensive answer from the following topk most relevant paragraphs and the given question. 
                                Provide a clear and concise response that summarizes the key points and information presented in the paragraphs. 
                                Your answer should be in your own words and be no longer than 50 words. \n\n Question: {query} \n\nParagraphs: {join(documents)}\n\n Answer:""",
                                output_parser=AnswerParser(),)
    elif template_name=='ynqa':
        prompt = PromptTemplate(
            name="ynqa",
            prompt_text="""Question: {query} \n\n Answer the question with either `Yes` or `No`.\n\n Paragraphs: {join(documents)} \n\nAnswer:""",
        )
    elif template_name=='emqa':
        prompt = PromptTemplate(
            name="emqa",
            prompt_text="""Question: {query} \n\n Answer the question by extracting the answer from the text. \n\Paragraphs: {join(documents)} \n\nAnswer:""",
        )
    else:
        raise ValueError(f"Prompt template {template_name} not found. Please choose one of the following: {default_prompts}")
    logger.info(f"Using prompt template: {prompt.name}")
    return prompt

def execute_pipeline(pipe:Pipeline, questions:list[str], file_path:str, **kwargs) -> list[str]:
    """Execute a generic retriever + model pipeline and save the results to a json file. 
    The key 'answers' is used for Extractive QA models, while the key 'results' is used for Generative QA models. 
    Returns the pipeline output."""
    if not os.path.exists(file_path): 
        answers = []
        for question in tqdm(questions):
            response = pipe.run(query=question, **kwargs) 
            answer = response.get('answers') if 'answers' in response else response.get('results')
            answer = answer[0] if isinstance(answer, list) else answer 
            answer = answer.answer if isinstance(answer, Answer) else answer
            answers.append(answer)
        logger.info(f"Saving results for {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(answers, f)
    else:
        logger.info(f"Results already exist for {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            answers = json.load(f)
    return answers

def evaluate_retriever(eval_dir:str, config:dict, retriever:BaseRetriever=None, 
                       eval_labels:list[MultiLabel]=None, params:dict=None) -> dict:
    "Evaluate retriever on the evaluation set. Evaluate each node in isolation"
    if not os.path.exists(eval_dir):
        assert retriever is not None, "Please provide a retriever to evaluate"
        assert eval_labels is not None, "Please provide evaluation labels"
        assert params is not None, "Please provide params"
        logger.info(f"Evaluating {config['retriever']['name']}...")
        pipe = DocumentSearchPipeline(retriever=retriever)
        eval_result = pipe.eval(
            labels=eval_labels, 
            params=params, 
            sas_model_name_or_path=config['evaluation']['sas_model_checkpoint'],
            add_isolated_node_eval=True, 
            custom_document_id_field="name" # use the document name as unique identifier for retriever evaluation -> P_ij subset D_i
        )
        eval_report = pipe.print_eval_report(eval_result, n_wrong_examples=0)
        logger.info(eval_report)
        os.makedirs(eval_dir, exist_ok=True)  
        with open(os.path.join(eval_dir, 'eval_report.txt'), 'w', encoding='utf-8') as f:
            f.write(str(eval_report))
        eval_result.save(eval_dir)   
    else:
        logger.info(f"Loading {config['retriever']['name']} evaluation results...")
        eval_result = EvaluationResult.load(load_dir=eval_dir)
    metrics = eval_result.calculate_metrics(
        simulated_top_k_retriever=config['retriever']['top_k'],
        document_scope="document_id_or_answer"
    )   
    return metrics['Retriever']

def evaluate_answers(answers:list[str], eval_labels:list[MultiLabel], 
                     sas_model_checkpoint:str='cross-encoder/stsb-roberta-large') -> pd.DataFrame:
    """Evaluate a list of answers (y_hat) against a list of evalation labels (y) in the form of MultiLabel objects. 
    Returns maximum value for each metric (F1, ROUGE, BLEU, EM, METEOR, SAS). 
    Returns a pandas DataFrame with the evaluation results (1 row per multilabel)."""
    if not all([isinstance(a, str) for a in answers]):
        raise TypeError(f"answers should be a list of str objects, not {type(answers)}")
    if not all([isinstance(a, MultiLabel) for a in eval_labels]):
        raise TypeError(f"eval_labels should be a list of MultiLabel objects, not {type(eval_labels)}")
    if len(answers)!=len(eval_labels):
        raise ValueError(f"answers and eval_labels should have the same length, not {len(answers)} and {len(eval_labels)}")
    # load metrics
    rouge_metric = load('rouge')
    bleu_metric = load('bleu')
    exact_match_metric = load('exact_match')
    meteor_metric = load('meteor')
    sas_model_metric = CrossEncoder(sas_model_checkpoint)
    results = pd.DataFrame()
    # iterate the eval labels + answers
    for idx, (y_hat, multi_label) in enumerate(zip(answers, eval_labels)):
        # get gold labels + predictions and normalize them
        gold_labels = [label.answer.answer for label in multi_label._labels]
        gold_labels_normalized = [normalize_answer(label) for label in gold_labels]
        predictions = [y_hat for i in range(len(gold_labels_normalized))]
        predictions_normalized = [normalize_answer(p) for p in predictions]
        # Calculate metrics
        f1 = pd.DataFrame([f1_score(p, r) for p, r in zip(predictions_normalized, gold_labels_normalized)], columns=['f1'])
        f1 = f1.loc[f1.f1.argmax(), :]
        rouge = pd.DataFrame([rouge_metric.compute(predictions=[p], references=[r]) 
                              for p, r in zip(predictions_normalized, gold_labels_normalized)])
        rouge = rouge.loc[rouge.rougeL.argmax(), :]
        exact_match = pd.DataFrame([exact_match_metric.compute(predictions=[p], references=[r]) 
                                    for p, r in zip(predictions_normalized, gold_labels_normalized)])
        exact_match = exact_match.loc[exact_match.exact_match.argmax(), :]
        try:
            bleu = pd.DataFrame([bleu_metric.compute(predictions=[p], references=[r]) 
                                 for p, r in zip(predictions_normalized, gold_labels_normalized)])
            bleu = bleu.loc[bleu.bleu.argmax(), :]
        except ZeroDivisionError:
            logger.error("ZeroDivisionError: bleu.compute() encountered an error.")
            bleu = pd.Series()
        meteor = pd.DataFrame([meteor_metric.compute(predictions=[p], references=[r]) 
                               for p, r in zip(predictions_normalized, gold_labels_normalized)])
        meteor = meteor.loc[meteor.meteor.argmax(), :]
        sas = pd.DataFrame([{'sas':float(sas_model_metric.predict([(p), (r)], show_progress_bar=False)), \
            'sas_model':sas_model_checkpoint} for p, r in zip(predictions_normalized, gold_labels_normalized)])
        sas = sas.loc[sas.sas.argmax(), :]
        # Concatenate results to a single record and append to results
        record = pd.concat([rouge, bleu, f1, exact_match, sas, meteor])
        record['idx'] = idx
        record['question'] = multi_label._query
        record['y_hat'] = y_hat
        record['y_hat_normalized'] = predictions_normalized[0]
        record['answers_gold'] = gold_labels
        record['answers_gold_normalized'] = gold_labels_normalized
        results = pd.concat([results, pd.DataFrame([record])]).reset_index(drop=True)
    return results

def evaluate_pipeline(retriever:BaseRetriever, model:BaseReader|BaseGenerator, retriever_name:str, model_name:str,
                      questions:list[str], eval_labels:list[MultiLabel], save_dir:str, 
                      generation_kwargs:dict, parameters:dict, sas_model_checkpoint:str='cross-encoder/stsb-roberta-large', 
                      cache_eval:bool=True) -> pd.DataFrame:
    "A function to evaluate a retriever + model pipeline and save the results to a json file."
    prompt = model.default_prompt_template if isinstance(model, PromptNode) else None
    logging.info        
    pipeline_hash = hash_md5(retriever_name + model_name + str(prompt) + str(generation_kwargs))
    eval_dir = os.path.join(save_dir, 'eval')
    file_path_eval = os.path.join(eval_dir, pipeline_hash + '.json')
    logger.info(f"Retriever: {retriever_name}, Model: {model_name}, Pipeline hash: {pipeline_hash}")
    if not os.path.exists(file_path_eval) or not cache_eval:
        file_path_pipe = os.path.join(save_dir, 'pipe', pipeline_hash + '.json')
        os.makedirs(os.path.dirname(file_path_pipe), exist_ok=True)
        logging.info(f"Executing pipeline...")
        pipe = Pipeline()
        pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
        pipe.add_node(component=model, name="Reader", inputs=["Retriever"])
        answers = execute_pipeline(pipe=pipe, questions=questions, file_path=file_path_pipe, params=parameters)
        evaluated = evaluate_answers(answers=answers, eval_labels=eval_labels, sas_model_checkpoint=sas_model_checkpoint)
        evaluated['retriever'] = retriever_name
        evaluated['model'] = model_name    
        evaluated['hash'] = pipeline_hash
        logger.info(f"Saving results to {file_path_eval}")
        os.makedirs(os.path.dirname(file_path_eval), exist_ok=True)
        evaluated.to_json(file_path_eval)
    else:
        logger.info(f"Loading evaluation results from {file_path_eval}")
        evaluated = pd.read_json(file_path_eval)
    return evaluated

def main(config:dict, devices:list) -> pd.DataFrame:
    "Main function to run the AWS Analysis. Returns a dataframe with the results."
    
    # Check validity of configuration
    assert config['pre_processing']['split']['size'] > config['pre_processing']['split']['stride'], \
        "The split size must be larger than the overlap size."
    assert config['pre_processing']['split']['stride'] >= 0, \
        "The overlap size must be larger than or equal to 0."

    # Create the directory to save the results
    logger.info(f"Starting ODQA system...")
    os.makedirs(config['data']['save_directory'], exist_ok=True)
        
    # AWS documentation repository
    logger.info(f"Attempting to connect to AWS documentation repository...")
    if not os.path.exists(config['data']['aws_directory']):
        logger.warning(f"Directory {config['data']['aws_directory']} does not exist. Cloning AWS documentation repository...")
        os.system(' '.join(['git', 'clone', config['data']['aws_documentation_repository'], config['data']['aws_directory']]))
    
    # import questions and answers
    logger.info(f"Importing questions and answers...")
    df_true = aws_docs_import_qa() 
    assert len(df_true)==100, f"Expected 100 questions, got {len(df_true)}"
    QUESTIONS = df_true['Question'].tolist()
    ANSWERS = df_true['Answer_Full'].tolist()
    DOCUMENTS = df_true['Document_True'].tolist()

    # Runtime parameters -> we make a seperate database index hash for every combination of parameters
    SETTINGS_HASH = config_hash(config)
    logging.basicConfig(filename=f"{config['data']['save_directory']}{SETTINGS_HASH}.log", level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s') 
    logger.info(f"SETTINGS_HASH: {SETTINGS_HASH}")  
        
    # Import the document store
    document_store, documents, labels = document_store_import(config=config, index_name=SETTINGS_HASH)
    #doc_sizes = [len(d.content.split()) for d in documents]
    
    # Retriever
    if config['retriever']['name']=='bm25':   
        retriever = BM25Retriever(document_store=document_store, top_k=config['retriever']['top_k'])
    elif config['retriever']['name']=='tfidf':
        retriever = TfidfRetriever(document_store=document_store, top_k=config['retriever']['top_k'])
    elif config['retriever']['name']=='embedding':
        retriever = EmbeddingRetriever(
            document_store=document_store, 
            embedding_model=config['document_store']['embedding_model'], 
            top_k=config['retriever']['top_k'])
    elif config['retriever']['name']=='dpr':
        retriever = DensePassageRetriever(document_store=document_store, 
                                          query_embedding_model=config['document_store']['dpr_query_embedding_model'], 
                                          passage_embedding_model=config['document_store']['dpr_passage_embedding_model'], 
                                          batch_size=config['pre_processing']['embeddings']['batch_size'], 
                                          use_fast_tokenizers=config['document_store']['use_fast_tokenizers'], 
                                          top_k=config['retriever']['top_k'], use_gpu=True, embed_title=True)
    else:
        raise ValueError(f"Retriever {config['retriever']['name']} not supported.")
            
    # Update embeddings
    if isinstance(retriever, DensePassageRetriever) or isinstance(retriever, EmbeddingRetriever):
        logger.info(f"Embedding count before: {embedding_count}")
        document_store.update_embeddings(retriever=retriever, 
                                         batch_size=config['pre_processing']['embeddings']['batch_size'], 
                                         update_existing_embeddings=config['pre_processing']['embeddings']['update_existing'])
        if config['document_store']['backend']=='faiss':
            document_store.save(config['document_store']['faiss']['index_path'], config['document_store']['faiss']['config_path'])
        embedding_count = document_store.get_embedding_count()
        logger.info(f"Embedding count after: {embedding_count}")
    
    # Check if labels are present     
    eval_labels = document_store.get_all_labels_aggregated(
        drop_negative_labels=config['evaluation']['drop_negative_labels'], 
        drop_no_answers=config['evaluation']['drop_no_answers'])
    assert len(eval_labels) > 0, "No labels found in evaluation data. Please check that labels are present in the evaluation data."
        
        # ANALYSIS
    PARAMETERS = {
        "Retriever": {
            "top_k": config['retriever']['top_k']
        }
    }    
    RUNTIME_HASH = hash_md5(SETTINGS_HASH + str(PARAMETERS))
    RUNTIME_DIR = os.path.join(config['data']['save_directory'], RUNTIME_HASH)
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(os.path.join(RUNTIME_DIR, f"config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Runtime hash: {RUNTIME_HASH}")
    
    # Plot the text distribution of the documents after pre-processing
    text_length = [len(d.content) for d in documents]    
    import matplotlib.pyplot as plt
    plt.hist(text_length, bins=100)
    plt.title(f'Text length distribution for {RUNTIME_HASH}\nmean: {np.mean(text_length):.0f}, median: {np.median(text_length):.0f}')
    plt.savefig(os.path.join(RUNTIME_DIR, 'text_distribution.png'))
    #plt.show()
        
    #Evaluate Retrievers using haystack 
    eval_dir = os.path.join(RUNTIME_DIR, config['retriever']['name'])
    retriever_metrics = evaluate_retriever(
        eval_dir=eval_dir,
        config=config,
        retriever=retriever,
        eval_labels=eval_labels,
        params=PARAMETERS,
    )
    logger.info(f"Retriever metrics: {retriever_metrics}")
    
    # Generator setup https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig
    PROMPT = get_prompt(template_name=config['generator']['prompt_template_name'])
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
    logger.info(f"Generation kwargs: {GENERATION_KWARGS}")
        
    # Run the pipeline for each reader/generator combination and save the results to disk using .json  
    df = pd.DataFrame()
    for model_type, models in config['models'].items():
        for model_name, model_name_or_path in models.items():
            if model_type=='reader':
                model = TransformersReader(model_name_or_path=model_name_or_path, devices=devices)
            elif model_type=='generator':
                api_key = get_key('./data/OPEN_AI_KEY.txt') if model_name_or_path in OPENAI_MODEL_NAMES else None
                model = PromptNode(model_name_or_path=model_name_or_path, default_prompt_template=PROMPT, model_kwargs=GENERATION_KWARGS, api_key=api_key, devices=devices)
            else:
                raise ValueError(f'Unknown model type {model_type}')
            evaluated = evaluate_pipeline(
                retriever=retriever, 
                model=model, 
                model_name=model_name,
                retriever_name=config['retriever']['name'], 
                questions=QUESTIONS,  
                eval_labels=eval_labels, 
                sas_model_checkpoint=config['evaluation']['sas_model_checkpoint'], 
                save_dir=RUNTIME_DIR, 
                parameters=PARAMETERS, 
                generation_kwargs=GENERATION_KWARGS,
                cache_eval=True,
            )
            df = pd.concat([df, evaluated])
            del model
            gc.collect()
    if len(df)==0:
        raise ValueError("No results were generated. Please check your configuration.")
    df.to_json(os.path.join(RUNTIME_DIR, 'results.json'), orient='records')
    
    # Aggregate results and save to disk        
    cols = ['retriever', 'model', 'rougeL', 'rouge1', 'rouge2', 'bleu', 'f1', 'exact_match', 'meteor', 'sas']
    df_metrics = df.loc[:, cols].groupby(['retriever', 'model']).mean().reset_index().round(4) 
    df_metrics.to_csv(os.path.join(RUNTIME_DIR, 'results_metrics.csv'), index=False)    
    df_metrics.columns = ['Retriever','Model','Rouge-L','Rouge-1','Rouge-2','BLEU', 'F1', 'EM', 'METEOR', 'SAS']
    multiply_by_100 = [ 'F1', 'EM', 'Rouge-L','Rouge-1','Rouge-2', 'BLEU', 'METEOR','SAS']
    df_metrics.loc[:, multiply_by_100] = df_metrics.loc[:, multiply_by_100] * 100 # Convert to percentage for some metrics
    df_metrics = df_metrics.round(2)
    df_metrics.to_csv(os.path.join(RUNTIME_DIR, 'results_metrics_clean.csv'), index=False)    
    logger.info(df_metrics)
    
    # Print results and save to .txt file
    out = ''
    for question in QUESTIONS:
        out += '-'*100 + '\n'
        out += f"\tQuestion: {question}\n"
        answers_gold = df.loc[df['question'] == question, 'answers_gold'].iloc[0]
        out += f'\tGold Answers: {answers_gold}\n'
        for idx, row in df.loc[df['question'] == question, :].iterrows():
            y_hat = row['y_hat'].replace('\n', ' ')
            out += f"Retriever: {row['retriever']}, Model:\t{row['model']}: {y_hat} (F1: {round(row['f1'], 2)} Exact Match: {row['exact_match']}, Rouge: {round(row['rougeL'], 3)}, SAS: {round(row['sas'], 3)})\n"
    with open(os.path.join(RUNTIME_DIR, "results.txt"), 'w') as f:
        f.write(out)
    
    # return the dataframe containing the results of the runtime
    logger.info(f"Done! Results can be found at {RUNTIME_DIR}")
    return df

def import_results(out_directory='./data/out/'):
    "Collect the results to a single dataframe for retriever + reader"
    dirs = [d for d in os.listdir(out_directory) if os.path.isdir(os.path.join(out_directory, d))]
    retriever, reader = [], []
    for hash in tqdm(dirs):
        # config
        try:
            logging.info(f"Importing results from {hash}")
            config = json.loads(open(os.path.join(out_directory, hash, 'config.json')).read())
            # retriever eval
            eval_dir = os.path.join(out_directory, hash, config['retriever']['name'])
            eval_retriever = evaluate_retriever(eval_dir = eval_dir, config=config)
            eval_retriever['name'] = config['retriever']['name']
            eval_retriever['split_size'] = int(config['pre_processing']['split']['size'])
            eval_retriever['stride'] = int(config['pre_processing']['split']['stride'])
            eval_retriever['top_k'] = int(config['retriever']['top_k'])
            retriever.append(eval_retriever)
            # reader eval
            result = pd.read_json(os.path.join(out_directory, hash, 'results.json'))
            result['split_size'] = int(config['pre_processing']['split']['size'])
            result['stride'] = int(config['pre_processing']['split']['stride'])
            result['top_k'] = int(config['retriever']['top_k'])
            result['pipeline_hash'] = result['hash']
            result['runtime_hash'] = hash
            del result['hash']
            reader.append(result)
        except:
            logger.error(f"Could not import {hash}")
    return {
        'retriever': pd.DataFrame(retriever),
        'reader': pd.concat(reader),
    }

# Runtime
if __name__ == '__main__':
    
    # logging
    logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    
    # Load default configuration
    config = config_import_yaml(file_path='main.yml')
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        prog='main.py', 
        description='ODQA system evaluation using Haystack.', 
        epilog='Text at the bottom of help',
        add_help=True)
    parser.add_argument('--split_size', 
                        help='The window size of documents', 
                        type=int)
    parser.add_argument('--split_stride', 
                        help='The stride size for splitting', 
                        type=int)
    parser.add_argument('--split_respect_boundary', 
                        help='Whether to respect boundary when splitting documents.', 
                        type=bool)
    parser.add_argument('--top_k', 
                        help="Top-k documents for retriever & model (reader/generator)", 
                        type=int)
    args = parser.parse_args() 
    logger.info(args)

    # Update configuration
    if args.split_size is not None:
        config['pre_processing']['split']['size'] = args.split_size 
    if args.split_stride is not None:
        config['pre_processing']['split']['stride'] = args.split_stride
    if args.split_respect_boundary is not None:
        config['pre_processing']['split']['respect_boundary'] = args.split_respect_boundary
    if args.top_k is not None:
        config['retriever']['top_k'] = args.top_k

    for k, v in config.items():
        logger.info(f"Configuration: {k}={v}")
    
    # NVIDIA GPU
    import torch
    devices = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    # AMD GPU
    # import torch_directml
    # devices =[torch_directml.device()]
     
    # Run main with configuration
    df = main(config=config, devices=devices)
    
    
