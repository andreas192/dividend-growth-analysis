FROM postgres:16-alpine

RUN apk add --no-cache python3 py3-pip \
    && pip3 install --no-cache-dir --break-system-packages \
        pandas psycopg2-binary

COPY docker/init/01-schema.sql /docker-entrypoint-initdb.d/01-schema.sql
COPY docker/init/02-load-data.sh /docker-entrypoint-initdb.d/02-load-data.sh
COPY docker/load_data.py /docker/load_data.py
COPY data/ /data/

RUN chmod +x /docker-entrypoint-initdb.d/02-load-data.sh

ENV DATA_DIR=/data
