from utils import get
BASE_URL = "https://huggingface.co"

# Tasks
def get_model_tasks(cache:bool=True) -> dict:
    soup = get(BASE_URL + "/models", local_directory='./data/bot_memory/', cache=cache)
    a = soup.find_all("a")
    names = [url.text.replace('\n', '').replace('\t', '').replace('\u200b', '') for url in a]
    urls = [url.get("href") for url in a]
    tag = {url.replace("/models?pipeline_tag=", '') : name for name, url in zip(names, urls) if url.startswith("/models?pipeline_tag")}
    return tag

def task_url(tag:str) -> str:
    url = BASE_URL + '/models?pipeline_tag=' + tag
    return url

def get_urls(url:str, cache:bool=False) -> list[str]:
    soup = get(url, local_directory='./data/bot_memory/', cache=cache)
    section = soup.find_all("section")
    urls = section[1].find_all("a")
    urls = [url.get("href") for url in urls]
    urls = [x[1:] for x in urls[1:]]
    return urls

# Models
def get_model_names(tag:str, page:int=1, sort:str="downloads") -> list[str]:
    task_url = BASE_URL + f"/models?pipeline_tag={tag}&page={page}&sort={sort}"
    urls = get_urls(task_url)
    return urls

def model_url(tag):
    url = BASE_URL + '/' + tag
    return url


# Datasets
# def get_dataset_names(task='question-answering', page = 1, sort="downloads", search=''):
#     ds_url = BASE_URL + f"/datasets?task_categories=task_categories:{task}&page={page}&sort={sort}&search={search}"
#     print(f"Navigating to {ds_url}")
#     return get_urls(ds_url)

# def dataset_url(dataset_name):
#     url = BASE_URL + '/datasets/' + dataset_name
#     return url
    

if __name__ == "__main__":
    tasks = get_model_tasks()
    models = get_model_names(tag='question-answering')
    models = get_model_names(tag='text2text-generation')
    models = get_model_names(tag='text-generation')