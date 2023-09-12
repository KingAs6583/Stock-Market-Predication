import yfinance as yf
import pandas as pd
from yahoo_fin.stock_info import tickers_nifty50
indices = {"Nifty_Fin":"NIFTY_FIN_SERVICE.NS","Nifty50":"^NSEI","Nifty_Metal":"^CNXMETAL","Nifty_IT":"^CNXIT","Dow_Jones":"^DJI"}
indexName = ("Nifty_Fin","Nifty50","Nifty_Metal","Nifty_IT","Dow_Jones")
stocks = ("INFY.NS","NMDC.NS","HCLTECH.NS","TATASTEEL.NS")
path = "upload_files/"


def indices_data_save(ticker,name,path):
    hcl = yf.Ticker(ticker)

    his = hcl.history(period="5y",actions=False)

    ticker = name+"_5y.csv"

    his.to_csv(path+ticker)

    csv = pd.read_csv(path+ticker)
    csv.round(decimals=2)
    csv['Date'] = pd.to_datetime(csv['Date']).dt.date
    csv.columns = map(str.lower, csv.columns)
    csv.to_csv(path+ticker, index=False)


def stock_data_save(ticker):
    hcl = yf.Ticker(ticker)

    his = hcl.history(period="5y",actions=False)

    ticker = ticker+"_5y.csv"

    his.to_csv(path+ticker)

    csv = pd.read_csv(path+ticker)
    csv.round(decimals=2)
    csv['Date'] = pd.to_datetime(csv['Date']).dt.date
    csv.columns = map(str.lower, csv.columns)
    csv.to_csv(path+ticker, index=False)

def refresh_indices():
    path = "Indices/"
    for x in indexName:
     indices_data_save(ticker=indices[x],name=x,path=path)

def refresh_nifty50():
    for x in tickers_nifty50():
        stock_data_save(ticker=x)

if __name__ == "__main__":
    refresh_nifty50()
    refresh_indices()