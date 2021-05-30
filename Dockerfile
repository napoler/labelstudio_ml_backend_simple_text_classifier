# FROM python:3.7
# FROM napoler/labelstudio_ml_backend_simple_text_classifier
# FROM heartexlabs/label-studio
# FROM fnndsc/ubuntu-python3
# FROM ubuntu
FROM continuumio/miniconda3


WORKDIR /tmp
COPY requirements.txt .
COPY environment.yml .
# # COPY sources.list /etc/apt/
# 激活环境
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate  label-studio-ml-backend" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN conda activate label-studio-ml-backend && apt-get update && apt install git -y  && conda install --yes --file requirements.txt

# RUN apt-get update && apt install git -y  && pip install --upgrade pip && pip install --no-cache  -r requirements.txt   uwsgi==2.0.19.1 supervisor==4.2.2  label-studio==1.0.1   git+https://github.com/heartexlabs/label-studio-ml-backend  
# -i http://pypi.douban.com/simple --trusted-host pypi.douban.com


# RUN useradd redis &&

# COPY uwsgi.ini /etc/uwsgi/
# COPY supervisord.conf /etc/supervisor/conf.d/


WORKDIR /app

COPY *.py /app/
#复制运行脚本
COPY entrypoint.sh /app/

EXPOSE 9090

# CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
# CMD [" label-studio-ml ", "start", "/app"]
ENTRYPOINT ["./entrypoint.sh"]
# ENTRYPOINT ["python", "run.py"]
# label-studio-ml start /app