# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /music-id
ADD . /music-id

RUN apt update && \
    apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git && \
    curl https://pyenv.run | bash && \
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc && \
    exec "$SHELL" && \
    pyenv install 3.6.5 && \
    pyenv virtualenv 3.6.5 music-id-py36 && \
    pyenv local music-id-py36 && \
    source ~/.bashrc && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* && apt clean
