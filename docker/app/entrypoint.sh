
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
        exec spark/spark-master 
fi

if [ $JOB == "5-wo" ]; then
        cd /app
        exec python3 -c "import simple_eval" # downloads models 
	exec spark/spark-worker  
fi

