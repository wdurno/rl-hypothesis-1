## configure
source config.sh
source get-job-id.sh
echo JOB_ID: ${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_JOB=k8s
export RL_HYPOTHESIS_1_INSTANCE=x${RL_HYPOTHESIS_1_JOB}-${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_DOCKER_IMAGE=${RL_HYPOTHESIS_1_DOCKER_CONTROLLER_IMAGE}
export RL_HYPOTHESIS_1_MACHINE_TYPE=e2-standard-2
## run 
source /app/scripts/spin-up-base-cluster.sh
