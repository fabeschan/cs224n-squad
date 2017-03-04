nvidia-docker run \
    -it \
    --workdir=/opt \
    -v `pwd`:/opt \
    keras $@
