export JOB=$1

source config.sh 
gcloud auth activate-service-account $RL_HYPOTHESIS_1_SERVICE_ACCOUNT_NAME --key-file service-account.json 
cat service-account.json | docker login -u _json_key --password-stdin https://gcr.io/gdax-dnn 

if [ $JOB == "4-bu" ]; then
	echo Attempting ai-base image build...
	cd /app/ai-base
	source docker-build.sh 
	echo Attempting ai image build...
	cd /app/ai
	source docker-build.sh 
fi
