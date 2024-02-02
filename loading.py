import json
import os
import helpers as hp
import pandas as pd
import firebase_admin as fb_ad
from firebase_admin import firestore
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_inputs(file_path: str):
    '''
    <font color="#f56e62">Load json file into a dictionary.</font>

    Parameters
    ===============
    **file_path** : <font color="#008001">str</font>
        <br>Path to the input file.

    Returns
    ===============

    **closed_mask** : <font color="#008001">dict</font>
        <br>Dictionary with the input parameters.
    '''
    with open(file_path, 'r') as json_file:
        inputs = json.load(json_file)
    return inputs


def firebase_setup(cred_path: str, verbose: bool = True):
    '''
    <font color="#f56e62">Initialise the firebase object.</font>

    Parameters
    ===============
    **cred_path** : <font color="#008001">str</font>
        <br>The path to the json file with the firebase credentials.

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)

    Returns
    ===============

    **db** : <font color="#008001">
    google.cloud.firestore_v1.client.Client
    </font>
        <br>The firestore client object.
    '''
    hp.log(f'Loading credentials from:{cred_path}', verbose)
    json_path = os.path.abspath(cred_path)
    cred = fb_ad.credentials.Certificate(json_path)
    fb_ad.initialize_app(cred)
    db = firestore.client()
    hp.log('Firestore client ready', verbose)

    return db


def load_news_from_db(ticker: str, db, path_base: str, verbose: bool = True):
    '''
    <font color="#f56e62">Pull all the news in a given (ticker),
      collection and save it as a dataframe (csv).</font>

    Parameters
    ===============

    **ticker** : <font color="#008001">str</font>
        <br>The ticker for the news

    **db** : <font color="#008001">
    google.cloud.firestore_v1.client.Client</font>
        <br>The firestore client object.

    **path_base**: <font color="#008001">str</font>
        <br>The path where to save the raw file.

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)

    Returns
    ===============

    **dataset** : <font color="#008001">pandas.core.frame.DataFrame</font>
        <br>DataFrame with all the news body,
          title and time associated to the given ticker.

    **csv_filename** : <font color="#008001">str</font>
        <br>Filename of the saved dataframe.

    **dataset_name** : <font color="#008001">str</font>
        <br>Name of the dataset.
        Unique id made by ticker name start_time and end_time.
    '''

    hp.log(f'creating dataset for: {ticker}', verbose)
    # navigating to the collection
    doc_ref = db.collection(f'news/{ticker}/raw_data')
    docs = doc_ref.stream()
    dataset = []
    times = list()
    bodies = list()
    titles = list()
    # reading the document
    for doc in docs:
        data = doc.to_dict()
        times.append(data['time'])
        titles.append(data['title'])
        bodies.append(data['body'])
    # make the dataset
    dataset = pd.DataFrame()
    dataset['time'] = times
    dataset['title'] = titles
    dataset['body'] = bodies
    # save the dataset
    df_start = times[0].strftime('%Y-%m-%d-%H-%M_')
    df_end = times[-1].strftime('%Y-%m-%d-%H-%M_')
    csv_filename = f'{path_base}/{ticker}_{df_start}{df_end}raw.csv'
    dataset.to_csv(csv_filename, index=False)
    dataset_name = f'{ticker}_{df_start}{df_end}'
    hp.log(f'dataset created, file saved at {csv_filename}', verbose)

    return dataset, csv_filename, dataset_name


def load_dataset_from_csv(path_dataset):
    '''
    <font color="#f56e62">lad dataset from csv file.</font>

    Parameters
    ===============
    **path_dataset** : <font color="#008001">str</font>
        <br>Path to the dataset file (csv).

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.
    '''
    df = pd.read_csv(path_dataset)
    return df


def load_model(model_name: str, verbose=True):
    '''
    <font color="#f56e62">Use the hugging face transformers
    library to load the model.</font>

    Parameters
    ===============

    **model_name** : <font color="#008001">str</font>
        <br>The model name from the the huggingface website:
          https://huggingface.co/models.

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)

    Returns
    ===============
    **tokenizer** : <font color="#008001">transformers.models.</font>
        <br>The text Tokenizer.

    **model** : <font color="#008001">transformers.models.</font>
        <br>The model that makes the prediction/generation.
    '''

    barts = ["BART", "bart"]
    for string in barts:
        if string in model_name:
            model = BertForSequenceClassification.from_pretrained(model_name,
                                                                  num_labels=3)
            tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name)
    hp.log(f'Model: {model_name} loaded', verbose)
    return tokenizer, model
