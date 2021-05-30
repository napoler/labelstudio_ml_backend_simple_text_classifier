# FROM python:3.7
# FROM napoler/labelstudio_ml_backend_simple_text_classifier
# FROM heartexlabs/label-studio
FROM fnndsc/ubuntu-python3

WORKDIR /tmp
COPY requirements.txt .

# # COPY sources.list /etc/apt/

RUN apt-get update &&  pip install --upgrade pip && pip install --no-cache  -r requirements.txt   uwsgi==2.0.19.1 supervisor==4.2.2  label-studio==1.0.1   git+https://github.com/heartexlabs/label-studio-ml-backend  
# -i http://pypi.douban.com/simple --trusted-host pypi.douban.com


# RUN useradd redis &&

# COPY uwsgi.ini /etc/uwsgi/
# COPY supervisord.conf /etc/supervisor/conf.d/


WORKDIR /app

COPY *.py /app/

EXPOSE 9090

# CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
CMD [" label-studio-ml ", "start", "/app"]

# label-studio-ml start /app