import os
import pandas as pd
import firebase_admin as fb_ad 
from firebase_admin import credentials , firestore
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np
from sklearn.linear_model import LinearRegression


def firebase_setup(cred_path:str, verbose:bool = True):
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

    **db** : <font color="#008001">google.cloud.firestore_v1.client.Client</font> 
        <br>The firestore client object.
    '''

    log(f'Loading credentials from:{cred_path}',verbose)
    json_path = os.path.abspath(cred_path)
    cred = fb_ad.credentials.Certificate(json_path)
    fb_ad.initialize_app(cred)
    db = firestore.client()
    log(f'Firestore client ready',verbose)

    return db


def get_news_from_db(ticker:str, db, verbose:bool=True):
    '''
    <font color="#f56e62">Pull all the news in a given (ticker) collection and save it as a dataframe (csv).</font> 
    
    Parameters
    ===============

    **ticker** : <font color="#008001">str</font> 
        <br>The ticker for the news

    **db** : <font color="#008001">google.cloud.firestore_v1.client.Client</font> 
        <br>The firestore client object.

    **verbose** : <font color="#008001">Bool</font> 
        <br>If True it enables logging (default True)

    Returns

    ===============

    **dataset** : <font color="#008001">pandas.core.frame.DataFrame</font> 
        <br>DataFrame with all the news body, title and time associated to the given ticker.
    '''

    log(f'creating dataset for: {ticker}', verbose)
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
    #save the dataset
    df_start = times[0].strftime('%Y-%m-%d-%H-%M_')
    df_end = times[-1].strftime('%Y-%m-%d-%H-%M_')
    csv_filename = f'data/{ticker}_{df_start}_{df_end}.csv'  
    dataset.to_csv(csv_filename, index=False)
    log(f'dataset created, file saved at {csv_filename}' , verbose)

    return dataset


def log(txt:str, verbose:bool): 
    '''
    <font color="#f56e62">format and print the text if verbose is true.</font> 
    
    Parameters
    ===============

    **txt** : <font color="#008001">str</font> 
        <br>The text that needs to be printed

    **verbose** : <font color="#008001">Bool</font> 
        <br>If True it enables logging (default True)
    '''
    if verbose: 
        print(f'----- {txt} -----')


def get_market_change(ticker, date, time_period, verbose=True):
    '''
    <font color="#f56e62">Pull the price of the given ticker and compute the change in the price over the time period.</font> 
    
    Parameters

    ===============

    **ticker** : <font color="#008001">str</font> 
        <br>The ticker for the news.

    **date** : <font color="#008001">str</font> 
        <br>The the start date.

    **time_period** : <font color="#008001">str</font> 
        <br>The time range over which the price change is computed.

    **verbose** : <font color="#008001">Bool</font> 
        <br>If True it enables logging (default True)

    Returns

    ===============

    **price_change** : <font color="#008001">list</font> 
        <br>Price variation over the selected period of time.
    '''
    
    stock = yf.Ticker(ticker)
    date_start = date
    date_end = date + pd.DateOffset(days=time_period)
    historical_data = stock.history(start=date_start, end=date_end, interval="1h")

    #TODO: 
    # This function has been rewritten as compute_market_change
    # delete it once compute_market_change is finished
    # Check if the following are done correctly 
    #  correct for: 
    #       USA remove SP500 (regress)
    #       CH remove SMI
    #       crypto remove BTC
    #       R_tesla = price - \beta R_sp500 

    price_change = historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-0]
    # make weighted average instead of future price - now price 

    return price_change


def compute_market_change(dataset, ticker, verbose=True):

    '''
    <font color="#f56e62"> Compute the adjusted future price for each element of the database.
      Add a column the dataframe with the adjusted future price and resaves the dataframe as a csv</font> 
    
    Parameters

    ===============

    **dataset** : <font color="#008001">pandas.core.frame.DataFrame</font> 
        <br>The DataFrame with the news and the date corresponding to each news.

    **ticker** : <font color="#008001">str</font> 
        <br>The ticker for the news.

    **verbose** : <font color="#008001">Bool</font> 
        <br>If True it enables logging (default True)

    '''

    log(f'Computing market change for: {ticker}' , verbose)

    dataset['time'] = pd.to_datetime(dataset['time'])

    df_start = dataset['time'].iloc[0].strftime('%Y-%m-%d-%H-%M_')
    df_end = dataset['time'].iloc[-1].strftime('%Y-%m-%d-%H-%M_')

    #dataset['market_change'] = dataset['time'].apply(lambda x: get_market_change(ticker, x, period))
    dataset['market_change'] = dataset['time'].apply(lambda x: correction_marker(x, ticker))
    csv_filename = f'data/{ticker}_{df_start}_{df_end}.csv'  
    dataset.to_csv(csv_filename, index=False)

    log(f'market change added, file saved at {csv_filename}' , verbose)





def load_model(model_name, verbose=True): 
    '''
    <font color="#f56e62">Use the hugging face transformers library to load the model.</font>

    Parameters
    ===============	
    **model_name** : <font color="#008001">str</font>
        <br>The model name from the the huggingface website: https://huggingface.co/models.

    **verbose** : <font color="#008001">Bool</font> 
        <br>If True it enables logging (default True)
    
    Returns
    ===============	
    **tokenizer** : <font color="#008001">transformers.models.</font>
        <br>The text Tokenizer.

    **model** : <font color="#008001">transformers.models.</font>
        <br>The model that makes the prediction/generation.    
    '''

    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return tokenizer, model

def make_single_prediction(tokenizer, model, text, verbose=True):
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results = nlp(text)
    
    
    return results[0]

def correction_marker(date, ticker, reg_period=30, target_period=7, verbose=True):

    date_time = pd.to_datetime(date)
    end_date = date_time + pd.DateOffset(days=target_period)
    spy_ohlc_df = yf.download('SPY', start=date_time, end=end_date)
    spy_close = np.array(spy_ohlc_df['Close'].values)
    hist_start_date = date_time - pd.DateOffset(days=reg_period)
    hist_data = yf.download(ticker, start=hist_start_date, end=date_time)
    x = np.arange(len(hist_data)).reshape(-1, 1)
    y = hist_data['Close'].values
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    # Get the beta coefficients
    intercept = model.intercept_
    slope = model.coef_[0]
    target_data =  yf.download(ticker, start=date_time, end=end_date)
    target_data_close = np.array(target_data['Close'].values)
    res = target_data_close - slope*spy_close
    step = 1 / target_period
    val= 0
    w = []
    for i in range(target_period):
        val = val + step
        w.append([val])
    res = target_data_close - slope*spy_close
    weighted_average = np.sum(res * w)
    return weighted_average



def load_adataset(path_dataset, verbose=True):
    df = pd.read_csv(path_dataset)
    return df



def make_prediction(tokenizer, model, dataset, text_col, model_name,ticker, verbose=True):
    '''
    <font color="#f56e62">Using a preloded model-tokenizer and given an imput text make a prediction.</font> 

    Parameters
    ===============
    **tokenizer** : <font color="#008001">transformers.models.bert.</font>
        <br>The text Tokenizer.
        
    **model** : <font color="#008001">transformers.models.bert.</font>
        <br>The model that makes the prediction/generation.

    **model_name** : <font color="#008001">str</font>
        <br>The text that we want to use as imput for the prediction.

    **verbose** : <font color="#008001">Bool</font> 
        <br>If True it enables logging (default True)
    
    Returns
    ===============
    **res** : <font color="#008001">dict</font> 
        <br>Dictionary containing the prediction 'label' and the 'score' associated  with the prediction.     
    '''

    dataset[f'preditction_{text_col}_{model_name}'] = dataset[text_col].apply(lambda x: make_single_prediction(tokenizer, model, x))
    df_start = dataset['time'].iloc[0].strftime('%Y-%m-%d-%H-%M_')
    df_end = dataset['time'].iloc[-1].strftime('%Y-%m-%d-%H-%M_')

    csv_filename = f'data/{ticker}_{df_start}_{df_end}.csv'  
    dataset.to_csv(csv_filename, index=False)

   

    """
    labels = []
    scores = []
    for r in results:
        labels.append(r["label"])
        scores.append(r["score"])
    res_df = pd.DataFrame()
    res_df["labels"] = labels
    res_df["score"] = scores

    res = {"label" :  res_df['labels'].value_counts().idxmax(),
           "score" : res_df['score'].mean()}

    """
    return 0


