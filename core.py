import loading as loader
import processing as processer
import saving as saver
import helpers as hp

# 0 - Read inputs
inputs_path = 'inputs.json'
inputs = loader.load_inputs(inputs_path)
cred_path = inputs["cred_path"]
ticker = inputs["ticker"]
path_base = inputs["base_path"]
period = inputs["period"]
max_lenght = inputs["max_lenght"]
col_lenght_name = inputs["col_lenght_name"]
model_name = inputs["model_name"]
nb_comp = inputs["nb_comp"]
nb_clust = inputs["nb_clust"]

# 1 - Load the ticker data from a db
db = loader.firebase_setup(cred_path)
dataset, path_df_raw, dataset_name = loader.load_news_from_db(ticker,
                                                              db,
                                                              path_base)
# 2 - Process the dataset
df = loader.load_dataset_from_csv(path_df_raw)
df = processer.remove_nan(df, path_base, dataset_name)
df = processer.compute_market_change(dataset, ticker, period)
df = processer.add_text_features(df)
df = processer.remove_long_text(df, max_lenght,
                                col_lenght_name,
                                path_base, dataset_name)

# 3 - Model label prediction
tokenizer, model = loader.load_model(model_name)
df = processer.make_prediction(tokenizer, model, df, 'title', model_name)
df = processer.make_prediction(tokenizer, model, df, 'body', model_name)

# 3.5 - Uncompress labels prediction
df = processer.split_prediction_labels(df, model_name)


for col in ['title', 'body']:
    # 4 - Sentence vectorization
    hp.log("vectorization", True)
    embs = processer.vectorization(df, col, path_base, dataset_name)

    # 5 - Dimensionality reduction
    hp.log("dimensionality reduction", True)
    df = processer.vectorization(df, embs, nb_comp, col)

    # 6 - Clustering
    print("clustering")
    df = processer.clustering(df, col, nb_clust, nb_comp)

# 7 - Save results
saver.save_dataset_as_csv(df, f"{path_base}/{dataset_name}_processed.csv")
