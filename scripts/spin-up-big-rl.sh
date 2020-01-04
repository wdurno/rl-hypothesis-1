IMAGE_NAME=${RL_HYPOTHESIS_1_DOCKER_REGISTRY_HEAD}rl-hypothesis-1-env:0.0.1 
VM_NAME=big-rl

gcloud beta compute --project=gdax-dnn instances create-with-container $VM_NAME \
	--zone=us-central1-a \
	--machine-type=n1-standard-16 \
	--subnet=default \
	--network-tier=PREMIUM \
	--metadata=google-logging-enabled=true \
	--maintenance-policy=MIGRATE \
	--no-service-account \
	--no-scopes \
	--image=cos-stable-78-12499-89-0 \
	--image-project=cos-cloud \
	--boot-disk-size=10GB \
	--boot-disk-type=pd-standard \
	--boot-disk-device-name=$VM_NAME \
	--container-image=$IMAGE_NAME \
	--container-restart-policy=always \
	--labels=container-vm=cos-stable-78-12499-89-0 \
	--reservation-affinity=any \
	--container-command=initial-rl.sh 

