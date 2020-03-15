## configure
export RL_HYPOTHESIS_1_JOB_ID=$(shuf -i 2000-65000 -n 1)
echo JOB_ID: ${RL_HYPOTHESIS_1_JOB_ID}
export RL_HYPOTHESIS_1_JOB=3-gan
export RL_HYPOTHESIS_1_INSTANCE=x${RL_HYPOTHESIS_1_JOB}-${RL_HYPOTHESIS_1_JOB_ID}
source config.sh
## run
source scripts/spin-up-vm.sh
