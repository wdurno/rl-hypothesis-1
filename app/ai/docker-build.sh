# build final image 
IMAGE_NAME=${RL_HYPOTHESIS_1_DOCKER_AI_IMAGE}
cat Dockerfile | envsubst | docker build . -t $IMAGE_NAME -f - --rm=false
docker push $IMAGE_NAME
