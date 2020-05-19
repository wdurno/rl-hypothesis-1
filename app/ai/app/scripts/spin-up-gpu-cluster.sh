gcloud beta container \
	--project "${RL_HYPOTHESIS_1_PROJECT}" \
	clusters create "${RL_HYPOTHESIS_1_INSTANCE}" \
	--zone "${RL_HYPOTHESIS_1_ZONE}" \
	--no-enable-basic-auth \
	--cluster-version "1.14.10-gke.27" \
	--machine-type "e2-small" \
	--image-type "COS" \
	--disk-type "pd-standard" \
	--disk-size "100" \
	--metadata disable-legacy-endpoints=true \
	--service-account "${RL_HYPOTHESIS_1_SERVICE_ACCOUNT_NAME}" \
	--num-nodes "1" \
	--enable-stackdriver-kubernetes \
	--enable-ip-alias \
	--network "${RL_HYPOTHESIS_1_NETWORK}" \
	--subnetwork "${RL_HYPOTHESIS_1_SUBNETWORK}" \
	--default-max-pods-per-node "110" \
	--no-enable-master-authorized-networks \
	--addons HorizontalPodAutoscaling,HttpLoadBalancing \
	--no-enable-autoupgrade \
	--enable-autorepair
gcloud beta container \
	--project "${RL_HYPOTHESIS_1_PROJECT}" \
	node-pools create "gpu-pool" \
	--cluster "${RL_HYPOTHESIS_1_INSTANCE}" \
	--zone "${RL_HYPOTHESIS_1_ZONE}" \
	--node-version "1.14.10-gke.27" \
	--machine-type "n1-standard-2" \
	--accelerator "type=nvidia-tesla-k80,count=1" \
	--image-type "COS" \
	--disk-type "pd-standard" \
	--disk-size "100" \
	--metadata disable-legacy-endpoints=true \
	--service-account "${RL_HYPOTHESIS_1_SERVICE_ACCOUNT_NAME}" \
	--num-nodes "1" \
	--no-enable-autoupgrade \
	--enable-autorepair
