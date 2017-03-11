docker rm run-project
nvidia-docker run \
    -it \
    --name run-project \
    --workdir=/opt \
    -v `pwd`:/opt \
    tensorflow/tensorflow:0.12.1-gpu $@
