import logging
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from haystack.schema import Document
import sqlite3

logger = logging.getLogger(__name__)

def read_stackexchange(file, condition:str='', skip_lines:int=2, nrows:int=1e3, block_size:int=1) -> pd.DataFrame:
    "Read a StackExchange XML file and parse it to a pandas dataframe."
    if block_size > nrows:
        block_size = nrows
    if condition == '':
        logging.warning(f'Condition is empty. This will severely slow down the parsing, and will crash the program if too many lines are read...')
    data = []
    with open(file, 'r') as f:
        # skip the first n lines
        logger.info(f'Skipping {skip_lines} lines...')
        for _ in tqdm(range(skip_lines)):
            __ = f.readline()
        # read the next n lines
        with tqdm(total=nrows, file=sys.stdout) as pbar:
            while nrows > 0:
                try:
                    line = f.readline()
                    nrows -= 1
                    if nrows % block_size == 0:
                        pbar.update(block_size)
                    data.append(line) if condition in line else None     
                except Exception as e:
                    logger.error(f'Error: {e}')
                    break
    # parse to pandas dataframe
    logger.info(f'Parsing {len(data)}')
    if len(data) > 1e5:
        logger.warning(f'Many lines to parse, this will take a while...')
    logger.info(f'Parsing xml to bs4.BeautifulSoup...')
    data = [BeautifulSoup(d, 'xml').find('row') for d in data]
    data = [d.attrs for d in data if d is not None]
    logger.info(f'Parsing to pd.DataFrame...')
    df = pd.DataFrame(data)
    return df

def get_stackexchange(db:str, query:str='SELECT * FROM posts;') -> pd.DataFrame:
    "Read stackexchange database to pandas dataframe."
    conn = sqlite3.connect(db)
    df = pd.read_sql(query, conn)
    df['Title'] = df['Title'].astype('str')
    df['Body'] = df['Body'].astype('str')
    df['Body_text'] = [BeautifulSoup(text, 'html.parser').get_text() for text in df['Body']]
    df['Tags'] = df['Tags'].astype('str')
    df['ClosedDate'] = df['ClosedDate'].astype('datetime64[ns]')
    df['Id'] = df['Id'].astype('int64')
    df['OwnerUserId'] = df['OwnerUserId'].astype('float64')
    df['ParentId'] = df['ParentId'].astype('float64')
    df['PostTypeId'] = df['PostTypeId'].astype('int64')
    # df['CreationDate'] = df['CreationDate'].astype('datetime64[ns]')
    # df['LastActivityDate'] = df['LastActivityDate'].astype('datetime64[ns]')
    # df['CommunityOwnedDate'] = df['CommunityOwnedDate'].astype('datetime64[ns]')
    df['ViewCount'] = df['ViewCount'].replace(np.NaN, 0).astype('int64')
    df['AnswerCount'] = df['AnswerCount'].replace(np.NaN, 0).astype('int64')
    df['CommentCount'] = df['CommentCount'].replace(np.NaN, 0).astype('int64')
    df['FavoriteCount'] = df['FavoriteCount'].replace(np.NaN, 0).astype('int64')
    df['LastEditorUserId'] = df['LastEditorUserId'].astype('float64')
    df['AcceptedAnswerId'] = df['AcceptedAnswerId'].astype('float64')
    df['Score'] = df['Score'].astype('int64')
    df['URL'] = 'https://stackoverflow.com/questions/' + df['Id'].astype('str')
    logger.info(f"Loaded {len(df)} records from {db}")    
    conn.close()
    return df

def stackexchange_documents(db:str, query:str='SELECT * FROM posts;', target_col:str='Body_text') -> list[Document]:
    "Read stackexchange database to list of `haystack.Document`."
    df = get_stackexchange(db=db, query=query)
    records = df.to_dict('records')
    return [Document(content = record.pop(target_col), meta=record) for record in records]
 

if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO)

    # Datasource: https://archive.org/details/stackexchange

    # Data description: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678

    # STACK EXCHANGE DOWNLOAD TO DF
    import os
    DATASET_LOC = '/mnt/p/datasets/stackoverflow/stackoverflow.com-Posts'
    DATASET_POSTS_LOC = os.path.join(DATASET_LOC, 'Posts.xml')
    df = read_stackexchange(file=DATASET_POSTS_LOC, condition='kubernetes', skip_lines=2, nrows=1e6, block_size=1)
    len(df)
    df
    
    # Import the data from the SQLite database
    df = get_stackexchange(db='./data/stackexchange_kubernetes.db')
    df.columns