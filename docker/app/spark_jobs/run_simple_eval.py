from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import json 

@udf('string')
def run_simple_eval_experiment(kwargs_json):
    from simple_eval import simple_eval_experiment 
    kwargs = json.loads(kwargs_json) 
    metric = simple_eval_experiment(**kwargs)['metric'] 
    metric_json = json.dumps(metric) 
    return metric_json 

test_args = spark.createDataFrame(
            [
                json.dumps({'sample_size': 10, 'probability_simulated': .5, 'metric_sample_size': 1}), 
                json.dumps({'sample_size': 30, 'probability_simulated': .5, 'metric_sample_size': 1}), 
                json.dumps({'sample_size': 50, 'probability_simulated': .5, 'metric_sample_size': 1}), 
                json.dumps({'sample_size': 70, 'probability_simulated': .5, 'metric_sample_size': 1}), 
                json.dumps({'sample_size': 100, 'probability_simulated': .5, 'metric_sample_size': 1}) 
            ],
            StringType() 
        )

def distributed_simple_experiment(args=test_args):
    return args.select(run_simple_eval_experiment('value')).show() 

