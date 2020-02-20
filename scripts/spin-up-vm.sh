gcloud beta compute --project=${RL_HYPOTHESIS_1_PROJECT} instances create-with-container ${RL_HYPOTHESIS_1_INSTANCE} \
	--zone=${RL_HYPOTHESIS_1_ZONE} \
	--machine-type=n1-standard-16 \
	--subnet=default \
	--network-tier=PREMIUM \
	--metadata=google-logging-enabled=true \
	--maintenance-policy=MIGRATE \
	--service-account=172005308156-compute@developer.gserviceaccount.com \
	--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
	--image=cos-stable-79-12607-80-0 \
	--image-project=cos-cloud \
	--boot-disk-size=300GB \
	--boot-disk-type=pd-standard \
	--boot-disk-device-name=${RL_HYPOTHESIS_1_INSTANCE} \
	--container-image=${RL_HYPOTHESIS_1_DOCKER_IMAGE} \
       	--container-restart-policy=always \
	--labels=container-vm=cos-stable-79-12607-80-0 \
	--container-env=JOB=${RL_HYPOTHESIS_1_JOB} \
        --container-env=ZONE=${RL_HYPOTHESIS_1_ZONE} \
	--container-env=PROJECT=${RL_HYPOTHESIS_1_PROJECT} \
	--container-env=INSTANCE=${RL_HYPOTHESIS_1_INSTANCE} 
