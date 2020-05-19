## configure
source config.sh 
if [ -z "$RL_HYPOTHESIS_1_JOB_ID" ]; then echo "Please get a job id"; exit 1; fi
echo JOB_ID: ${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_JOB=3-bu
export RL_HYPOTHESIS_1_INSTANCE=x${RL_HYPOTHESIS_1_JOB}-${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_DOCKER_IMAGE=${RL_HYPOTHESIS_1_DOCKER_CONTROLLER_IMAGE}
export RL_HYPOTHESIS_1_MACHINE_TYPE=e2-standard-2
## run 
cat app/ai/app/kubernetes/builder-pod.yaml | envsubst | kubectl apply -f -
