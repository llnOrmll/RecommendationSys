from utils import krx_api
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from tqdm import tqdm
import time
import os
root_dir = '~/PycharmProjects/recommend'
df_sise = pd.read_excel(os.path.join(root_dir, 'datasets/krx_stock_sise.xlsx'))
df_info = pd.read_excel(os.path.join(root_dir, 'datasets/krx_stock_info.xlsx'))
df_all = pd.merge(df_sise, df_info[['단축코드', '주식종류']], left_on='종목코드', right_on='단축코드', how='inner')

df_all = df_all[df_all['주식종류'] == '보통주']
df_all.drop('주식종류', axis=1, inplace=True)
df_all.shape

df_filter = df_all[['종목코드','종목명','시장구분','종가','거래대금','시가총액','상장주식수']].copy()
df_filter.columns = ['code','name','mkt_segment','close','transaction_volume','mrk_cap','shares_outstanding']
df_filter.info()

market_str = df_filter['mkt_segment'].unique()
market_num = [2, 1, 3]
df_filter['mkt_segment'].replace(market_str, market_num, inplace=True)
df_filter = df_filter[df_filter['mkt_segment'] != 3]

#Description
def get_description(array: np.array):
    if array[1] == 1:
        url = 'https://finance.yahoo.com/quote/' + array[0] + '.KS/profile?p=' + array[0] + '.KS'
    else:
        url = 'https://finance.yahoo.com/quote/' + array[0] + '.KQ/profile?p=' + array[0] + '.KQ'
    resp = requests.get(url)
    time.sleep(2)
    if resp.status_code == 200:
        try:
            soup = BeautifulSoup(resp.content, 'html.parser')
            desc = soup.findAll('p', {'class': 'Mt(15px) Lh(1.6)'})[0].text
        except Exception:
            desc = ''
        return desc
    else:
        print("Network is down")

array_list = df_filter[['code', 'mkt_segment']].values
array_list_split = np.array_split(array_list, 2)

with Pool(4) as p:
    start = time.time()
    results = list(tqdm(p.imap(get_description, array_list_split[1]), total=len(array_list_split[1])))
    print("Time elapsed: {}".format(time.time() - start))

r1 = results
r2 = results
r_all = r1+r2
df_filter['desc'] = r_all
df_filter.to_csv('datasets/stock_dataset.csv')

###########################################

df = pd.read_csv('datasets/stock_dataset.csv')
res_all = pd.read_csv('datasets/res_all.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

df['desc'] = df['desc'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['desc'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(name, cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    stock_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[stock_indices]

get_recommendations('JW생명과학')



#Sector
def create_url(code):
    return "https://wisefn.finance.daum.net/v1/company/c1010001.aspx?cmp_cd=" + code
df_filter['url'] = df_filter['code'].apply(create_url)

def get_sector(url):
    res = requests.get(url)
    time.sleep(2)
    if res.status_code == 200:
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.findAll('table', {'id': 'comInfo'})
        wics = table[0].findChildren('span', {'class': 'exp'})[1].text.split()[-1]
        return wics
    else:
        return ''

def add_feature(df):
    df['sector'] = df['url'].apply(get_sector)
    return df

def parall_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    p = Pool(n_cores)
    df = pd.concat(tqdm(p.map(func, df_split)))
    p.close()
    p.join()
    return df

df_split = np.array_split(df_filter, 200)
df_split_1 = df_split[:100]
df_split_2 = df_split[100:]

#res_list = []
#with Pool(4) as p:
#    with tqdm(total=len(df_split)) as pbar:
#        for i, res in enumerate(p.imap_unordered(add_feature, df_split)):
#            pbar.update()
#            res_list.append(res)

with Pool(4) as p:
    start = time.time()
    results = list(tqdm(p.imap(add_feature, df_split_2), total=len(df_split_2)))
    print("Time elapsed: {}".format(time.time() - start))
res_all_2 = pd.concat(results)

res_all = pd.concat([res_all_1, res_all_2], axis=0)
res_all.to_csv(os.path.join(root_dir, "datasets/res_all.csv"))

#Stats
df_filter['cat'] = pd.cut(df_filter['close'], bins=[0,5000,10000,20000,50000,100000,1000000], labels=[
    '<5000','<10000','<20000','<50000','<100000','<1000000'
])
cat_counts = df_filter['cat'].value_counts()
cat_counts

plt.bar(cat_counts.index, cat_counts.values)
plt.xlabel('Price range')
plt.ylabel('Counts')
plt.show()

#
df_below_lv1 = df_filter[df_filter['cat'] == '<5000']
df_below_lv1.describe()

df_below_lv1.sort_values(by='transaction_volume', ascending=False)[['name', 'transaction_volume']].head(30)

