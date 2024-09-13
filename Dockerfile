FROM quay.io/jupyter/minimal-notebook:latest

USER root
RUN apt-get -qq update && apt-get install -y --no-install-recommends apt-utils openssh-client

USER $NB_UID

WORKDIR /opt/busanalysis/

ADD requirements.txt .
RUN pip install -r requirements.txt

CMD  jupyter lab \
        --ip=0.0.0.0 \
        --port=8085 \
        --allow-root \
        --NotebookApp.notebook_dir='./notebooks' \
        --NotebookApp.token='' \
        --NotebookApp.password=''