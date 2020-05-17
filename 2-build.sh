## configure
source config.sh 
export RL_HYPOTHESIS_1_JOB_ID=${RL_HYPOTHESIS_1_WORKFLOW_ID}-$(shuf -i 2000-65000 -n 1)
echo JOB_ID: ${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_JOB=2-bu
export RL_HYPOTHESIS_1_INSTANCE=x${RL_HYPOTHESIS_1_JOB}-${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_DOCKER_IMAGE=${RL_HYPOTHESIS_1_DOCKER_CONTROLLER_IMAGE}
export RL_HYPOTHESIS_1_MACHINE_TYPE=e2-standard-2
## run 
source scripts/spin-up-build-cluster.sh
cat kubernetes/builder-pod.yaml | envsubst | kubectl apply -f -
