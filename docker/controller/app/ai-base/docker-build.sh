# build final image 
IMAGE_NAME=${RL_HYPOTHESIS_1_DOCKER_AI_BASE_IMAGE}
docker build . -t $IMAGE_NAME --build-arg BUCKET_NAME=$RL_HYPOTHESIS_1_BUCKET --rm=false
docker push $IMAGE_NAME
