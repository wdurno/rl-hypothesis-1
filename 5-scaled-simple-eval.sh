## configure
export RL_HYPOTHESIS_1_JOB_ID=$(shuf -i 2000-65000 -n 1)
echo JOB_ID: ${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_JOB=5-k8
export RL_HYPOTHESIS_1_INSTANCE=x${RL_HYPOTHESIS_1_JOB}-${RL_HYPOTHESIS_1_JOB_ID}
source config.sh 
# TODO login to kubernetes
scripts/spin-up-cluster.sh 
# apply envs and deploy 
cat kubernetes/spark-master-deployment.yaml | envsubst | kubectl apply -f - 
cat kubernetes/spark-master-service.yaml | envsubst | kubectl apply -f -
cat kubernetes/spark-worker-deployment.yaml | envsubst | kubectl apply -f -
