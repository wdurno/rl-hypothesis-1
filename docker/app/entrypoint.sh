
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

