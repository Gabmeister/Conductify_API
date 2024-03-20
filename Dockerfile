FROM python:3.10-slim-buster
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV SPOTIFY_CLIENT_ID=9ada1ae6a8154663a48f889d10cf8faf
ENV SPOTIFY_CLIENT_SECRET=12e1f946e15c415f99a72bfa233193ce
ENV SPOTIFY_REDIRECT_URI=conductifyhgr://callback
CMD gunicorn --bind 0.0.0.0:$PORT app:app