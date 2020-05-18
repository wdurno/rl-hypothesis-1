FROM  ${RL_HYPOTHESIS_1_DOCKER_AI_BASE_IMAGE}

ADD app /app
ADD /app/spark/spark-defaults.conf /opt/spark/conf/spark-defaults.conf
ENV PATH $PATH:/opt/spark/bin 
RUN chmod 777 /app/entrypoint.sh 

WORKDIR /app

ENTRYPOINT ["bash", "/app/entrypoint.sh"]

