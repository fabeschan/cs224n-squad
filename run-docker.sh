nvidia-docker run \
    -it \
    --workdir=/opt \
    -v `pwd`:/opt \
    tensorflow/tensorflow:0.12.1-gpu $@
