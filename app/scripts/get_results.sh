source ../config.sh
gsutil cp gs://${RL_HYPOTHESIS_1_BUCKET}/experimental-result.pkl /tmp/experimental-result.pkl 
python3 get_results.py

