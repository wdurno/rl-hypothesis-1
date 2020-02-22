
# directory may already exist as K8s volume mount 
mkdir -p /dat

if [ $JOB == "0-de" ]; then
        while true; do echo sleeping; sleep 100; done;  
fi

if [ $JOB == "1-rl" ]; then
	cd /app
	exec python3 rl.py
fi

if [ $JOB == "2-tr" ]; then
        cd /app
        exec python3 transfer_sampler.py
fi

if [ $JOB == "3-gan" ]; then
	cd /app
	exec python3 cgan.py
fi 

if [ $JOB == "4-si" ]; then
        cd /app
        exec python3 simple_eval.py
fi

if [ $JOB == "5-ma" ]; then
        cd /app
        spark/spark-master 
fi

if [ $JOB == "5-wo" ]; then
        cd /app
        python3 get_models.py 
	spark/spark-worker 
fi

