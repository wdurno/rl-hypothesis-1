
if [ $JOB == "1-RL" ]; then
	cd /app
	exec python3 rl.py
fi

if [ $JOB == "2-TR" ]; then
        cd /app
        exec python3 transfer_sampler.py
fi

if [ $JOB == "3-GAN" ]; then
	cd /app
	exec python3 cgan.py
fi 

