## configure
export RL_HYPOTHESIS_1_JOB_ID=$(shuf -i 2000-65000 -n 1)
echo JOB_ID: ${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_JOB=5-k8
export RL_HYPOTHESIS_1_INSTANCE=x${RL_HYPOTHESIS_1_JOB}-${RL_HYPOTHESIS_1_JOB_ID}
source config.sh 
## spin-up cluster 
scripts/spin-up-cluster.sh 
## apply envs and deploy 
cat kubernetes/spark-master-deployment.yaml | envsubst | kubectl apply -f - 
cat kubernetes/spark-master-service.yaml | envsubst | kubectl apply -f -
cat kubernetes/spark-worker-deployment.yaml | envsubst | kubectl apply -f -
# wait until ready
! STATE=$(kubectl get pods | tail -n+2 | grep master | awk '{print $3;}')
if [ -z $STATE ]; then STATE=NOT_READY; fi
while [ $STATE != "Running" ]; do 
	! STATE=$(kubectl get pods | tail -n+2 | grep master | awk '{print $3;}') 
	if [ -z $STATE ]; then STATE=NOT_READY; fi
	echo waiting for master to be ready...
	sleep 5 
done
## get master 
MASTER=$(kubectl get pods | tail -n+2 | grep master | awk '{print $1;}')
## enable background process execution 
set -m 
## run job 
kubectl exec $MASTER bash /app/run_spark_simple_eval.sh & 
