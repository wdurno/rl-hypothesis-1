
echo "configuring environment variables..."

echo "===="
cat config.sh 
echo "===="
source config.sh  
echo "===="
env | grep RL_HYPOTHESIS_1  
echo "===="

echo "building image..."

echo "===="
cd docker/ai
cp $RL_HYPOTHESIS_1_SERVICE_ACCOUNT_JSON_PATH app/service-account.json 
cat docker-build.sh 
echo "===="
bash docker-build.sh 
echo "===="
rm app/service-account.json

