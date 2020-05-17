
echo "configuring environment variables..."

echo "===="
cat config.sh 
echo "===="
source config.sh  
echo "===="
env | grep RL_HYPOTHESIS_1  
echo "===="

echo "building controller image..."

echo "===="
cp config.sh docker/controller/app/config.sh
cp $RL_HYPOTHESIS_1_SERVICE_ACCOUNT_JSON_PATH docker/controller/app/service-account.json 
cd docker/controller
cat docker-build.sh 
echo "===="
bash docker-build.sh 
echo "===="
rm app/service-account.json

