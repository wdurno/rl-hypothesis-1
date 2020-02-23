import json 

def run_simple_eval_experiment(kwargs_json_iterable):
    from simple_eval import simple_eval_experiment # RAM and disk heavy 
    metrics = [] 
    for kwargs_json in kwargs_json_iterable: 
        kwargs = json.loads(kwargs_json) 
        metric = simple_eval_experiment(**kwargs)['metric'] # compute heavy 
        metrics.append(metric) 
    return iter(metrics) 

test_args = [
    json.dumps({'sample_size': 10, 'probability_simulated': .5, 'metric_sample_size': 1}), 
    json.dumps({'sample_size': 30, 'probability_simulated': .5, 'metric_sample_size': 1}), 
    json.dumps({'sample_size': 50, 'probability_simulated': .5, 'metric_sample_size': 1}), 
    json.dumps({'sample_size': 70, 'probability_simulated': .5, 'metric_sample_size': 1}), 
    json.dumps({'sample_size': 100, 'probability_simulated': .5, 'metric_sample_size': 1}) 
]
test_args = sc.parallelize(test_args, 2) 

def distributed_simple_experiment(args=test_args):
    return args.mapPartitions(run_simple_eval_experiment).collect()  

#distributed_simple_experiment()

