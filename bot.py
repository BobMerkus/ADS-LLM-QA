# Libraries
import logging
import os
import pickle
import pandas as pd
from datetime import datetime

from transformers import pipeline
from haystack.schema import Document
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline

# custom modules
from utils import get_key, get_stackexchange
from utils.nlp import normalize_answer

class ChatBot():
    """Generates a bot object that can talk.
    The `self.conversations` attribute stores the conversations in a dictionary, where the key is the conversation id and the value is a list of messages.
    Disk  Management (write to disk and update metadata around class) 
    Conversation Management (save, pop, append, messages)
    API
    """

    def __init__(self, id:str='New Conversation', directory:str='./data/bot_memory/', name:str='ChatBot V1.0') -> None:
        self.directory = directory
        self.id = id
        self.name = name
        self.conversations = dict() 
        self.read()
        self.__update__()

    def __repr__(self) -> str:
        return f"<ChatBot with {self.__len__()} messages ({self.shape})>)"
    
    def __shape__(self) -> tuple:
        return tuple(len(c) for c in self.conversations.values())
    
    def __len__(self) -> int:
        return sum(self.shape)
    
    def __update__(self) -> None:
        "If the conversation does not exist, create it. Update the metadata of the class."
        if 'New Conversation' not in self.conversations.keys():
            self.conversations['New Conversation'] = [] 
        self.shape = self.__shape__()
        self.n_messages = self.__len__()
    
    def write(self) -> None:
        "Write conversations to pickle"
        self.__update__()
        with open('./data/conversations.pickle', 'wb') as f:
            pickle.dump(self.conversations, f)
        
    def read(self) -> None:
        "Read conversations from pickle"
        if os.path.exists('./data/conversations.pickle'):
            with open('./data/conversations.pickle', 'rb') as f:
                self.conversations = pickle.load(f)
        self.__update__()
    
    def save(self, id:str):
        "If the current conversation is a new one, rename it to the new id"
        self.conversations[id] = self.conversations.pop(self.id)
        self.id = id
        self.write()
    
    def pop(self, id:str=None, message_dt:str=None) -> list:
        "Remove a conversation, if none is specified, remove the current"
        if message_dt:
            message_index = [i for i, msg in enumerate(self.conversations[self.id]) if msg['datetime']==message_dt][0]
            logging.info(f"Removed message {message_dt} ({message_index}) from conversation {self.id}")
            removed = self.conversations[self.id].pop(message_index)
        else:
            if id is None:
                id = self.id
            removed = self.conversations.pop(id)
            logging.info(f"Removed conversation {id}")
        self.id = 'New Conversation'
        self.write()
        return removed
        
    def append(self, user, data) -> dict:
        "Append a message to the current conversation"
        new_data = {'datetime' : str(datetime.now()), 'user' : user, 'data' : data}
        if self.id not in self.conversations.keys():
            self.conversations[self.id] = []
        self.conversations[self.id].append(new_data)
        self.write()
        return new_data
    
    def messages(self, id:str=None, type:type=None) -> list:
        "Return the messages of a certain type"
        if id is not None:
            self.id = id 
        conversation = self.conversations[self.id]
        if type is not None:
            if type==pd.DataFrame:
                conversation = pd.DataFrame(conversation)
            elif type==str:
                conversation = [msg['data'] if isinstance(msg['data'], str) else '' for msg in conversation]
        return conversation
        
    def response(self, question:str, top_k:int=5, context:str|list[str]='', answer:str|list[str]='', user_name:str='Anonymous') -> dict:
        "The purpose of the response is to generate a response to the question."
        if (question is None or question=='') and (context is None or context=='' or context==[] or context==['']):
            logging.error('Either a Question or Context has to be supplied')
            return None
        # Normalize the answer   
        if isinstance(answer, str):
            answer = [answer]
        # Add user message
        user_message = {'user':user_name, 'datetime':datetime.now(), 'data':(question, context)}
        self.append(user=user_name, data=user_message)
        result = {
            #'question':question,
            'answer_correct':answer,
            'bot_class':str(self.__class__),
            'bot_name':self.name,
            'user_name':user_name,
            'top_k':top_k,
            'answer':None
        }
        #self.append(user=BOT_NAME, data=result)
        #self.write() 
        return result

class PipelineChatbot(ChatBot):
    def __init__(self, id: str = 'New Conversation', directory: str = './data/bot_memory/', name: str = 'ChatBot V1.0') -> None:
        super().__init__(id, directory, name)
        self.model_checkpoint = None
        self.pipe = None
        
class HuggingfaceChatBot(PipelineChatbot):
    
    def pipeline(self, model_checkpoint:str, task:str='text-generation', *args, **kwargs):
        "Set the pipeline"
        self.model_checkpoint = model_checkpoint
        self.pipe = pipeline(task=task, model=self.model_checkpoint, *args, **kwargs)

class HaystackChatBot(PipelineChatbot):
    
    def pipeline(self, model_checkpoint:str):
        prompt = PromptTemplate(name="lfqa",
            prompt_text="""Synthesize a comprehensive answer from the following topk most relevant paragraphs and the given question. 
            Provide a clear and concise response that summarizes the key points and information presented in the paragraphs. 
            Your answer should be in your own words and be no longer than 50 words. 
            \n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:""",
            output_parser=AnswerParser(),)
        self.pipe = PromptNode(model_name_or_path=model_checkpoint, use_gpu=True, default_prompt_template=prompt)
        
    def response(self, question:str, top_k:int=5, n:int=5, context:str|list[str]='', answer:str|list[str]='', user_name:str='Anonymous', *args, **kwargs) -> dict:
        "The purpose is to generate a response to the question. This includes handling the pipeline initialization and conversation management."
        if not self.pipe:
            raise RuntimeError(f"Initiate a pipeline first")
        result = super().response(question=question, top_k=top_k, n=n, context=context, answer=answer, user_name=user_name)
        if kwargs=={}:
            logging.warning('Did you forget to specify the `kwargs`? Most downstream methods require some specific parameters.')
        response = self.pipe.run(query=question, top_k=top_k, context=context, *args, **kwargs)
        result = {**result, **response}
        # add bot message + write to disk
        self.append(user=self.name, data=result)
        self.write() 
        return result
    
if __name__=="__main__":
    
    # set debug level 
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

        # Closed Book QA
    #Load the bot
    b = ChatBot()
    b.shape
    b.messages()
    b.save(id='Test')
    b.pop(id='Test')
    b.messages()
    b.conversations.keys()
    
    # b_huggingface = ChatBot(retriever=Retriever(), reader=HuggingFaceReader())
    # # Ask a question based on context
    # answer = b_huggingface.response(question='Who Invented the Internet?',n=5, answer='vinton cerf and bob kahn', 
    #                                 reader_kwargs={'task':'question-answering', 'model_checkpoint':"distilbert-base-uncased-distilled-squad"},
    #                                 context='What most of us think of as the Internet is really just the pretty face of the operation—browser windows, websites, URLs, and search bars. But the real Internet, the brain behind the information superhighway, is an intricate set of protocols and rules that someone had to develop before we could get to the World Wide Web. Computer scientists Vinton Cerf and Bob Kahn are credited with inventing the Internet communication protocols we use today and the system referred to as the Internet')    
    # print(answer)
    # # Ask a question based on context from a pdf
    # answer = b_huggingface.response(question='What does Question Answering aim to provide?', context='https://arxiv.org/pdf/2101.00774.pdf', 
    #                                 reader_kwargs={'task':'question-answering', 'model_checkpoint':"distilbert-base-uncased-distilled-squad"})    
    # print(answer['answer'], answer['score'])
    # # use the t5 model for qa        
    # answer = b_huggingface.response(question='Who Invented the Internet?', reader_kwargs={'task':'text2text-generation', 'model_checkpoint':"google/flan-t5-base"}) # with context is not working
    # answer = b_huggingface.response(question='Who Invented the Internet?', reader_kwargs={'task':'text2text-generation', 'model_checkpoint':"google/flan-t5-base"},
    #                                 context='What most of us think of as the Internet is really just the pretty face of the operation—browser windows, websites, URLs, and search bars. But the real Internet, the brain behind the information superhighway, is an intricate set of protocols and rules that someone had to develop before we could get to the World Wide Web. Computer scientists Vinton Cerf and Bob Kahn are credited with inventing the Internet communication protocols we use today and the system referred to as the Internet')
    # print(answer['answer'])




    # # Kubernetes QA
    # df_kubernetes = pd.concat([pd.read_json('./data/kubernetes_docs.json'), pd.read_json('./data/kubernetes_blog.json')])
    # r_kubernetes = VectorSpaceModelRetriever(df=df_kubernetes, column_name='text')
    # r_kubernetes.get(question='What is a pod?', n=5)
    # b_kubernetes = ChatBot(retriever=r_kubernetes, reader=HuggingFaceReader())
    # b_kubernetes.response(question='What is a pod?', n=5, 
    #                       reader_kwargs={'task':'question-answering', 'model_checkpoint':"distilbert-base-uncased-distilled-squad"})
    # b_kubernetes.response(question='What is the command to get the logs of a pod?', n=5, 
    #                       reader_kwargs={'task':'question-answering', 'model_checkpoint':"distilbert-base-uncased-distilled-squad"})
    # b_kubernetes.response(question='What is the command to get the logs of a pod?', n=5, 
    #                         reader_kwargs={'task':'text2text-generation', 'model_checkpoint':"google/flan-t5-base"})
    
    # # STACK EXCHANGE QA
    # df_stackexchange = get_stackexchange(db=STACKEXCHANGE_DB, query='SELECT * FROM posts;')
    # r_stackexchange = VectorSpaceModelRetriever(df=df_stackexchange, column_name='Title')
    # b_stackexchange = ChatBot(retriever=r_stackexchange, reader=HuggingFaceReader())
    # b_stackexchange.response(question='What is the difference between Kubernetes and Google Cloud Platform?', n=5, 
    #                          reader_kwargs={'task':'text2text-generation', 'model_checkpoint':"google/flan-t5-base"})

    # # StackExchange QA based on the body
    # r_stackexchange = VectorSpaceModelRetriever(df=df_stackexchange, column_name='Body')
    # b_stackexchange = ChatBot(retriever=r_stackexchange, reader=HuggingFaceReader())
    # b_stackexchange.response(question='What is the difference between Kubernetes and Google Cloud Platform?', n=5, 
    #                          reader_kwargs={'task':'text2text-generation', 'model_checkpoint':"google/flan-t5-base"})

    # # Bing QA
    # r_bing = BingRetriever()
    # r_bing.get(question='What is the difference between Kubernetes and Google Cloud Platform?', n=5)

    # r_bing.memory



    # TRIVIA QA
    from datasets import load_dataset
    trivia_qa = load_dataset('trivia_qa', 'rc')
    trivia_qa['train'][0].keys()
    question = trivia_qa['train'][0]['question']
    context = trivia_qa['train'][0]['search_results']['search_context']
    answer = trivia_qa['train'][0]['answer']['aliases']
    answer = trivia_qa['train'][0]['answer']['normalized_aliases']
    result = b.response(task='question-answering', model_checkpoint="distilbert-base-uncased-distilled-squad", question=question, context=context, answer=answer)
    result['answer']
    result['rouge']
