FROM continuumio/anaconda3
RUN mkdir production
COPY . /production
EXPOSE 5000
WORKDIR /production
RUN pip install -r requirements.txt
CMD python flask_production.py