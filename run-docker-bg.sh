docker rm run-project
nvidia-docker run \
    -d \
    --name run-project \
    --workdir=/opt \
    -v `pwd`:/opt \
    -v /tmp:/tmp \
    tfimg $@
