gcloud beta container \
	--project "gdax-dnn" \
	clusters create "${RL_HYPOTHESIS_1_INSTANCE}" \
	--zone "us-central1-a" \
	--no-enable-basic-auth \
	--cluster-version "1.14.10-gke.17" \
	--machine-type "n1-standard-4" \
	--image-type "COS" \
	--disk-type "pd-standard" \
	--disk-size "100" \
	--metadata disable-legacy-endpoints=true \
	--scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
	--num-nodes "1" \
	--enable-stackdriver-kubernetes \
	--enable-ip-alias \
	--network "projects/gdax-dnn/global/networks/default" \
	--subnetwork "projects/gdax-dnn/regions/us-central1/subnetworks/default" \
	--default-max-pods-per-node "110" \
	--addons HorizontalPodAutoscaling,HttpLoadBalancing \
	--no-enable-autoupgrade \
	--enable-autorepair \
	--enable-autoupgrade 
gcloud beta container --project "gdax-dnn" \
	node-pools create "spark-worker" \
	--cluster "${RL_HYPOTHESIS_1_INSTANCE}" \
	--zone "us-central1-a" \
	--node-version "1.14.10-gke.17" \
	--machine-type "n1-highmem-4" \
	--image-type "COS" \
	--disk-type "pd-standard" \
	--disk-size "100" \
	--metadata disable-legacy-endpoints=true \
	--scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
	--preemptible \
	--num-nodes "${RL_HYPOTHESIS_1_N_WORKERS}" \
        --enable-autorepair \
	--enable-autoupgrade 
