
# enable detach with "&" 
set -m 

# directory may already exist as K8s volume mount 
mkdir -p /dat

if [ $JOB == "0-de" ]; then
        while true; do echo sleeping; sleep 100; done;  
fi

if [ $JOB == "1-rl" ]; then
	cd /app
	exec python3 -u rl.py
fi

if [ $JOB == "2-tr" ]; then
        cd /app
        exec python3 -u transfer_sampler.py
fi

if [ $JOB == "3-vae" ]; then
	cd /app
	exec python3 -u cvae.py
fi 

if [ $JOB == "4-si" ]; then
        cd /app
        exec python3 -u simple_eval.py
fi

if [ $JOB == "5-ma" ]; then
        cd /app
	# run master in background 
        spark/spark-master &
	# give master service time to start 
	sleep 3
       	# start job 
	spark-submit spark_simple_eval.py
fi

if [ $JOB == "5-wo" ]; then
        cd /app
        python3 -u get_models.py 
	spark/spark-worker 
fi

