# 
FROM python:3.9

# 
WORKDIR /code

# 
# COPY ./services/backend/requirements.txt /code/requirements.txt

# 
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt 

# 
COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
