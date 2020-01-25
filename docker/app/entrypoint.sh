
if [ $JOB == "1-RL" ]; then
	cd /app/rl
	python3 rl.py
fi

if [ $JOB == "2-GAN" ]; then
	cd /app/gan
	python3 cgan.py
fi 

