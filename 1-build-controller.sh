
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
cp config.sh app/config.sh
cp config.sh app/ai/app/config.sh 
cp $RL_HYPOTHESIS_1_SERVICE_ACCOUNT_JSON_PATH app/service-account.json
cp $RL_HYPOTHESIS_1_SERVICE_ACCOUNT_JSON_PATH app/ai/app/service-account.json 
cat docker-build.sh 
echo "===="
bash docker-build.sh 
echo "===="
rm app/service-account.json
rm app/ai/app/service-account.json 
rm app/config.sh
rm app/ai/app/config.sh 
