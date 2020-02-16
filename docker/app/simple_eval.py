
# Before running more-advanced experiments, I want to test for feasibility. 
# In an advanced evaluation, sampling should reflect the automaton's 
# experience, with the GAN being trained on only the observed real data. 
# In this simple evaluation, the GAN will not be trained--I need to 
# justify the engineering work first. The GAN is trained on a single, 
# large dataset. The evaluation will work as follows.
# 1. Sample data, pooling simulants with real. 
# 2. Fit the q-net 
# 3. Generate the metric: %-wins of game trials 

def simple_sample(sample_size, probability_simulated): 
    '''
    Generates a mixed dataset of simulated and real embedded samples. 
    Samples are "embedded" because we've used transfer learning. 
    Sampling is "simple" because the GAN is not fit with each simple. 
    '''
    ## sample real data 
    ## sample fake data 
    ## merge and return 
    pass

def fit(data):
    '''
    Fits a transfer-learned model on embedded data. Once fit, the 
    abstract model is combined with its lower parts (ie. convolutions) 
    and returned. Metrics are also returned to support later diagnostics. 
    '''
    ## Fit q-net on data 
    ## Combine with lower transfer-learned layers
    ## Return model and metrics 
    pass

def metric_trials(model, sample_size): 
    '''
    Metric: Percent-win of games per model. 
    '''
    pass

