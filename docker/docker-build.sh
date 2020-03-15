# build final image 
IMAGE_NAME=${RL_HYPOTHESIS_1_DOCKER_REGISTRY_HEAD}rl-hypothesis-1-env:1.0.0 
docker build . -t $IMAGE_NAME --build-arg BUCKET_NAME=$RL_HYPOTHESIS_1_BUCKET -f Dockerfile  
docker push $IMAGE_NAME
echo "image built and pushed: $IMAGE_NAME" 
