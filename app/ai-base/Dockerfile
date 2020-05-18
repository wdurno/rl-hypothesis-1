#FROM jaimeps/rl-gym:latest
# local copy 
FROM gcr.io/gdax-dnn/rl-base@sha256:9acd3a449e9460a150379a03c43e7a59d56cd0f82d7d4a2eeb8c4965fa31de56

ARG BUCKET_NAME
ENV BUCKET_NAME=${BUCKET_NAME} \
    HADOOP_VERSION=2.7.3 \
    SPARK_VERSION=2.2.1 \
    PYSPARK_PYTHON=python3 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/cuda/lib64

# cudnn drivers 
COPY cudnn-8.0-linux-x64-v6.0.tgz /cudnn-8.0-linux-x64-v6.0.tgz

# ml dependencies 
RUN pip install scikit-image && \
pip install keras==2.2.5 && \
pip install google-cloud-storage && \
pip install google-api-python-client && \
pip install oauth2client && \
apt-get update && \
apt-get install vim -y && \
apt-get install openjdk-8-jdk -y && \
# gpu 
apt-get install software-properties-common -y && \
DEBIAN_FRONTEND=noninteractive apt-get install lightdm -y && \
apt install wget && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && \
dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && \
apt-get update && \
apt-get -y install cuda && \
tar -zxvf cudnn-8.0-linux-x64-v6.0.tgz && \
rm cudnn-8.0-linux-x64-v6.0.tgz && \
pip uninstall tensorflow -y && \
pip install tensorflow-gpu==1.4.1 && \
# hadoop 
mkdir -p /opt && \
cd /opt && \
curl http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | \
    tar -zx hadoop-${HADOOP_VERSION}/lib/native && \
ln -s hadoop-${HADOOP_VERSION} hadoop && \
echo Hadoop ${HADOOP_VERSION} native libraries installed in /opt/hadoop/lib/native && \
# spark 
mkdir -p /opt && \
cd /opt && \
curl http://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz | \
    tar -zx && \
ln -s spark-${SPARK_VERSION}-bin-hadoop2.7 spark && \
echo Spark ${SPARK_VERSION} installed in /opt

