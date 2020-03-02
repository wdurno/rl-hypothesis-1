from pyspark import SparkContext
sc = SparkContext()

from gcp_api_wrapper import upload_blob, delete_cluster  
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

def study_probability_axis(n_real_fake=[(1000, 1000), (1000, 2000)], metric_sample_size=100):
    args = []
    for rf_pair in n_real_fake:
        for _ in range(metric_sample_size): 
            args.append(json.dumps({'n_real': rf_pair[0], 'n_fake': rf_pair[1], 'metric_sample_size': 1})) 
    return args

test_args = study_probability_axis(n_real_fake=[
                (1000, 0), 
                (1000, 1000),
                (1000, 10000), 
                (1000, 300000), 
                (100000, 0), 
                (100000, 1000), 
                (100000, 10000), 
                (100000, 100000), 
                (100000, 300000),
                (100000, 600000) 
            ]
        ) 
test_args = sc.parallelize(test_args, len(test_args)) 

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
    print('job done, shutting down cluster...') 
    delete_cluster() 
    sleep(100) 
