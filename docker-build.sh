# build final image 
IMAGE_NAME=${RL_HYPOTHESIS_1_DOCKER_CONTROLLER_IMAGE}
docker build . -t $IMAGE_NAME --rm=false 
docker push $IMAGE_NAME
