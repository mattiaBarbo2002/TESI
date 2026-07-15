FROM python:3.10-slim
ARG USER=rsde
ARG USER_ID=1006 # uid from the previus step
ARG USER_GROUP=rsde
ARG USER_GROUP_ID=1006 # gid from the previus step
ARG USER_HOME=/home/${USER}
# create a user group and a user (this works only for debian based images)
RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER
# setup image istructions
RUN apt-get update && apt-get install -y curl libexpat1
# set container user
USER $USER
# run script as non-root user
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
