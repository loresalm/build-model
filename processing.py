import ast
import torch
import pandas as pd
import helpers as hp
import yfinance as yf
import numpy as np
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans


def remove_nan(dataset, path_base, dataset_name):
    '''
    <font color="#f56e62">Remove the row containing nan.
        discarted rows are saved in the LOG_nan.csv file .</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **path_base** : <font color="#008001">str</font>
        <br>Path to where the data will be stored.

    **dataset_name** : <font color="#008001">str</font>
        <br>Name of the dataset.
        Unique id made by ticker name start_time and end_time.

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe without nan.
    '''
    df = dataset.copy()
    dropped_rows = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        nan_rows = df[df[column].isna()]
        dropped_rows = pd.concat([dropped_rows, nan_rows])
    dropped_rows.to_csv(f"{path_base}/{dataset_name}_LOG_nan.csv", index=False)
    df = df.dropna()
    return df


def correction_marker(date, ticker, reg_period=30, target_period=7):
    '''
    <font color="#f56e62"> Compute the adjusted future price
      for a given date</font>

    Parameters
    ===============

    **date** : <font color="#008001">str</font>
        <br>The current date for the correction.

    **ticker** : <font color="#008001">str</font>
        <br>The ticker for the news.

    **reg_period** : <font color="#008001">int</font>
        <br>The backtrack period for the regression in days.

    **target_period** : <font color="#008001">int</font>
        <br>How many days we want to look into the future.

    Returns
    ===============

    **weighted_average** : <font color="#008001">float</font>
        <br>Weighted_average of the market change over the selected period.

    '''
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
    slope = model.coef_[0]
    target_data = yf.download(ticker, start=date_time, end=end_date)
    target_data_close = np.array(target_data['Close'].values)
    res = target_data_close - slope*spy_close
    step = 1 / target_period
    val = 0
    w = []
    for i in range(target_period):
        val = val + step
        w.append([val])
    res = target_data_close - slope*spy_close
    weighted_average = np.sum(res * w)
    return weighted_average


def compute_market_change(dataset, ticker: str, verbose: bool = True):
    '''
    <font color="#f56e62"> Compute the adjusted future price
      for each element of the database.
      Add a column the dataframe with the adjusted future price
      and resaves the dataframe as a csv</font>

    Parameters
    ===============

    **dataset** : <font color="#008001">pandas.core.frame.DataFrame</font>
        <br>The DataFrame with the news and the date corresponding
        to each news.

    **ticker** : <font color="#008001">str</font>
        <br>The ticker for the news.

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)

    Returns
    ===============

    **dataset** : <font color="#008001">pandas.core.frame.DataFrame</font>
        <br>DataFrame with added column containing the market change.

    '''

    hp.log(f'Computing market change for: {ticker}', verbose)
    # get the column with times and start and end for file naming
    dataset['time'] = pd.to_datetime(dataset['time'])
    # apply correction for each element
    dataset['market_change'] = dataset['time'].apply(
        lambda x: correction_marker(x, ticker))
    hp.log('market change added', verbose)
    return dataset


def add_text_features(dataset):
    '''
    <font color="#f56e62">Add features to the dataset:
     - sign: if the market change is positive or negative
     - body_Length: number of char in the body text
     - title_Length: number of char in the title text.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the added columns.
    '''

    df = dataset.copy()
    df['sign'] = np.where(df['market_change'] >= 0, 0, 1)
    df['body_Length'] = df["body"].apply(len)
    df['title_Length'] = df["title"].apply(len)
    return df


def remove_long_text(dataset, max_lenght, column, path_base, dataset_name):
    '''
    <font color="#f56e62">Remove the rows with text that is too long.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **max_lenght** : <font color="#008001">int</font>
        <br>Max lenght allowed.

    **column** : <font color="#008001">str</font>
        <br>Name of the column containing the lenght data.

    **path_base** : <font color="#008001">str</font>
        <br>Path to where the data will be stored.

    **dataset_name** : <font color="#008001">str</font>
        <br>Name of the dataset.
        Unique id made by ticker name start_time and end_time.

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the added columns.
    '''

    df = dataset
    long_body = df[df[column] > max_lenght]
    long_body.to_csv(f"{path_base}/{dataset_name}_LOG_longBody.csv",
                     index=False)
    df = df[df[column] <= max_lenght]
    return df


def make_single_prediction(tokenizer, model, text: str, verbose=True):
    '''
    <font color="#f56e62">Use the hugging face transformers library
      to load the model.</font>

    Parameters
    ===============

    **tokenizer** : <font color="#008001">transformers.models.</font>
        <br>The text Tokenizer.

    **model** : <font color="#008001">transformers.models.</font>
        <br>The model that makes the prediction/generation.

    **text** : <font color="#008001">str</font>
        <br>The string of text to be analized.

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)

    Returns
    ===============

    **results** : <font color="#008001">tuple</font>
        <br> the ouput form the model in the form of (label, score)
    '''

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    if isinstance(text, str):
        results = nlp(text)[0]
    else:
        print(type(text))
        print(text)
        results = pd.NA
    return results


def make_prediction(tokenizer, model, dataset, text_col, model_name,
                    verbose=True):
    '''
    <font color="#f56e62">Using a preloded model-tokenizer and given
      an imput text make a prediction.</font>

    Parameters
    ===============
    **tokenizer** : <font color="#008001">transformers.models.bert.</font>
        <br>The text Tokenizer.

    **model** : <font color="#008001">transformers.models.bert.</font>
        <br>The model that makes the prediction/generation.

    **model_name** : <font color="#008001">str</font>
        <br>Name of the model.

    **verbose** : <font color="#008001">Bool</font>
        <br>If True it enables logging (default True)

    Returns
    ===============
    **res** : <font color="#008001">dict</font>
        <br>Dictionary containing the prediction 'label' and the 'score'
          associated  with the prediction.
    '''

    dataset[f'preditction_{text_col}_{model_name}'] = dataset[text_col].apply(
        lambda x: make_single_prediction(tokenizer, model, x))
    hp.log("text classification done", verbose)
    return dataset


def get_score_lab(text):
    '''
    <font color="#f56e62">Extract from a string the label and the score.</font>

    Parameters
    ===============
    **text** : <font color="#008001">str</font>
        <br>String that needs to be parsed.
        For example "label: Neutral, score: 0.95"

    Returns
    ===============
    **val** : <font color="#008001">List</font>
        <br>list containing label and score.
    '''
    try:
        txt_dict = ast.literal_eval(text)
        val = [txt_dict['label'], txt_dict['score']]
    except ValueError as e:
        print(f"Error: {e}")
        val = ["error", 1.0]

    return val


def split_prediction_labels(dataset, model_name):
    '''
    <font color="#f56e62">Remove the rows with text that is too long.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **model_name** : <font color="#008001">str</font>
        <br>Name of the model.

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the added columns contining the score and the label.
    '''
    df = dataset.copy()
    for tag in ["title", "body"]:
        col_name = f"preditction_{tag}_{model_name}"
        col_lab = f'label_{tag}_{model_name}'
        col_score = f'score_{tag}_{model_name}'
        df[[col_lab, col_score]] = df[col_name].apply(
                                        lambda x: pd.Series(get_score_lab(x)))
    return df


def mean_pooling(model_output, attention_mask):
    '''
    <font color="#f56e62">Apply mean poling to the embddings.</font>

    Parameters
    ===============
    **model_output** : <font color="#008001">np.array</font>
        <br>Array with the ebeddings.

    **attention_mask** : <font color="#008001">np.array</font>
        <br>Attention mask from the encoded input.

    Returns
    ===============

    **mean_pool** : <font color="#008001">np.array</font>
        <br>Mean pool of the embeddings.
    '''
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    mean_pool = torch.sum(
        token_embeddings*input_mask_expanded, 1)/torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
    return mean_pool


def sentence_embeddings(sentences):
    '''
    <font color="#f56e62">Generate embeddings from text.
    the embeddings are saved as a csv file.</font>

    Parameters
    ===============
    **sentences** : <font color="#008001">np.array</font>
        <br>Array containing sentences that need to be vectorized.

    Returns
    ===============

    **sentence_embeddings** : <font color="#008001">np.array</font>
        <br>. Array containing the embeddings,
        each row of the array corresponds to a row in the sentences array
    '''
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L12-v2')
    model = AutoModel.from_pretrained(
        'sentence-transformers/all-MiniLM-L12-v2')
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output,
                                       encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def vectorization(dataset, col, path_base, dataset_name):
    '''
    <font color="#f56e62">Generate embeddings from text.
    the embeddings are saved as a csv file.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **col** : <font color="#008001">str</font>
        <br>Name of the column that needs to be vectorized.

    **path_base** : <font color="#008001">str</font>
        <br>Path to where the data will be stored.

    **dataset_name** : <font color="#008001">str</font>
        <br>Name of the model.

    Returns
    ===============

    **embs** : <font color="#008001">np.array</font>
        <br>Embeddings corresponding to each sentence in the selected col.
    '''
    df = dataset.copy()
    sents = df[col].values.tolist()
    embs = sentence_embeddings(sents).numpy()
    np.savetxt(f"{path_base}/{dataset_name}_emb_{col}.csv",
               embs, delimiter=',')
    return embs


def dimensionality_reduction(dataset, embs, nb_comp, col):
    '''
    <font color="#f56e62">reduce the dimensionality of the embeddings.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **embs** : <font color="#008001">np.array</font>
        <br>Embeddings corresponding to each sentence in the selected col.

    **nb_comp** : <font color="#008001">int</font>
        <br>number of components (dimensions of the final space).

    **col** : <font color="#008001">str</font>
        <br>Name of the column "body" or "title".

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the added columns
          contining value for each component.
    '''
    df = dataset.copy()
    pca = PCA(n_components=nb_comp)
    reduced_data = pca.fit_transform(embs)
    for c in range(nb_comp):
        df[f'PC{c}_{col}'] = reduced_data[:, c]
    umap = UMAP(n_components=nb_comp)
    reduced_embeddings = umap.fit_transform(embs)
    for c in range(nb_comp):
        df[f'UMAP{c}_{col}'] = reduced_embeddings[:, c]
    return df


def clustering(dataset, col, nb_clust, nb_comp):
    '''
    <font color="#f56e62">perform clustering over the embeddings.</font>

    Parameters
    ===============
    **dataset** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the dataset.

    **col** : <font color="#008001">str</font>
        <br>Name of the column "body" or "title".

    **nb_clust** : <font color="#008001">int</font>
        <br>number of clusters.

    **nb_comp** : <font color="#008001">int</font>
        <br>number of components (dimensions of the final space).

    Returns
    ===============

    **df** : <font color="#008001">pd.DataFrame</font>
        <br>Dataframe with the added columns
          contining the cluster label.
    '''
    df = dataset.copy()
    for red in ['PC', 'UMAP']:
        kmeans = KMeans(n_clusters=nb_clust)
        select_cols = []
        for c in range(nb_comp):
            select_cols.append(f'{red}{c}_{col}')
        df[f'cluster_labels_{red}_{col}'] = kmeans.fit_predict(df[select_cols])
    return df
