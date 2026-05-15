FROM nvcr.io/nvidia/pytorch:24.01-py3

# Copy the requirements file in the app directory to avoid rebuilding the image
COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Fix the error in Opendataval
COPY ./odv_fix/_metrics.py /usr/local/lib/python3.10/dist-packages/opendataval/metrics.py
COPY ./odv_fix/_fetcher.py /usr/local/lib/python3.10/dist-packages/opendataval/dataloader/fetcher.py
COPY ./odv_fix/_exp_api.py /usr/local/lib/python3.10/dist-packages/opendataval/experiment/api.py
COPY ./odv_fix/_model_api.py /usr/local/lib/python3.10/dist-packages/opendataval/model/api.py

