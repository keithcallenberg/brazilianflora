FROM python:3.6.9-slim

#RUN mkdir -p /app
RUN apt-get update
RUN apt-get -y install libjpeg-dev libpng-dev libtiff-dev
RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get -y install libxvidcore-dev libx264-dev
RUN apt-get -y install libgtk-3-dev
WORKDIR /app

#ADD requirement text file to docker GOT STUCK xD BEFORE FIGURING OUT
COPY Logistic.pickle /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install --upgrade pip

#RUN ON LOCAL HOST 5000 address
EXPOSE 5000

# Add Model code
COPY MyModel.py /app/

# Define environment variables

ENV MODEL_NAME MyModel
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
