from pyspark import SparkContext
sc = SparkContext()

from gcp_api_wrapper import upload_blob 
from time import sleep 
import pickle 
import json 
import os 

# constants 
N_WORKERS = os.environ['N_WORKERS'] 
RESULT_PATH = 'experimental-result.pkl'
RESULT_BLOB_NAME = 'experimental-result.pkl' 

def run_simple_eval_experiment(kwargs_json_iterable):
    from simple_eval import simple_eval_experiment # RAM and disk heavy 
    metrics = [] 
    for kwargs_json in kwargs_json_iterable: 
        kwargs = json.loads(kwargs_json) 
        metric = simple_eval_experiment(**kwargs) # compute heavy 
        metrics.append(metric) 
    return iter(metrics) 

example_args = [
    json.dumps({'sample_size': 10000, 'probability_simulated': .1, 'metric_sample_size': 1}), 
    json.dumps({'sample_size': 10000, 'probability_simulated': .2, 'metric_sample_size': 1}),
    json.dumps({'sample_size': 10000, 'probability_simulated': .3, 'metric_sample_size': 1}),
    json.dumps({'sample_size': 10000, 'probability_simulated': .4, 'metric_sample_size': 1}),
    json.dumps({'sample_size': 10000, 'probability_simulated': .7, 'metric_sample_size': 1}),
    json.dumps({'sample_size': 10000, 'probability_simulated': .9, 'metric_sample_size': 1})  
]

def study_probability_axis(probability_simulated=[.1,.5,.9], sample_size=10000, metric_sample_size=1):
    args = []
    for p in probability_simulated:
        args.append(json.dumps({'sample_size': sample_size, 'probability_simulated': p, 'metric_sample_size': metric_sample_size}))
    return args

test_args = study_probability_axis(probability_simulated=[.0, .1, .2, .3, .4, .5, .6, .7, .8 , .9, 1.]*2) 
test_args = sc.parallelize(test_args, 11) 

def distributed_simple_experiment(args=test_args):
    return args.mapPartitions(run_simple_eval_experiment).collect()  

# execute experiment 
results = distributed_simple_experiment(test_args)

# save results 
output = list(zip(test_args.collect(), results)) 
with open(RESULT_PATH, 'wb') as f:
    pickle.dump(output, f)

upload_blob(RESULT_PATH, RESULT_BLOB_NAME) 

while True:
    # TODO shutdown cluster 
    print('job done') 
    sleep(100) 
