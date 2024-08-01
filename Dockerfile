FROM jupyter/pyspark-notebook:spark-3.2.1

USER root
RUN apt-get -qq update && apt-get install -y --no-install-recommends apt-utils openssh-client

USER $NB_UID

WORKDIR /opt/busanalisys/

ADD requirements.txt .
RUN pip install -r requirements.txt

CMD python setup.py -q develop && jupyter lab \
        --ip=0.0.0.0 \
        --port=8085 \
        --allow-root \
        --NotebookApp.notebook_dir='./notebooks' \
        --NotebookApp.token='' \
        --NotebookApp.password=''
