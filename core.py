import helpers as hp 


cred_path = "cred.json"


db = hp.firebase_setup(cred_path)
ticker = "AMZN"


dataset = hp.get_news_from_db(ticker, db)
"""

period = 10 
hp.compute_market_change(dataset, ticker, period)

model_name= "ahmedrachid/FinancialBERT-Sentiment-Analysis"
tk, md = hp.load_model(model_name)

hp.make_prediction(tk, md, dataset, 'title', model_name, ticker)
"""



path_dataset = "data/AMZN_2023-10-10-16-03__2023-10-09-12-01_.csv"
df = hp.load_adataset(path_dataset)
#hp.compute_market_change(df, ticker, 10)

time = df['time'].iloc[0]
price = df['market_change'].iloc[0]


hp.correction_marker(time, ticker)



