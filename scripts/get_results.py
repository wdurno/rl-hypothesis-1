import pickle
import json
import pandas as pd

with open('/tmp/experimental-result.pkl', 'rb') as f:
    x = pickle.load(f)

n_real = [json.loads(y[0].decode())['n_real'] for y in x]
n_fake = [json.loads(y[0].decode())['n_fake'] for y in x]
metric = [float(y[1]) for y in x]
df = pd.DataFrame({'n_real': n_real, 'n_fake': n_fake, 'metric': metric}) 
print(df.shape)
print(df.groupby(['n_real', 'n_fake']).agg(['mean', 'std'])) 

