
echo "configuring environment variables..."

echo "===="
cat config.sh 
echo "===="
source config.sh  
echo "===="
env | grep RL_HYPOTHESIS_1  
echo "===="

echo "building base image..."

echo "===="
cd docker/ai-base
cat docker-build.sh 
echo "===="
bash docker-build.sh 

