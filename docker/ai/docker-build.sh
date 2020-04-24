# build final image 
IMAGE_NAME=${RL_HYPOTHESIS_1_DOCKER_REGISTRY_HEAD}rl-hypothesis-1-env:ai-2.0.0 
cat Dockerfile | envsubst | docker build . -t $IMAGE_NAME -f -
docker push $IMAGE_NAME
echo "image built and pushed: $IMAGE_NAME" 
