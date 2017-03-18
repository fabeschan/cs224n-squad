docker rm run-project
nvidia-docker run \
    -it \
    --name run-project \
    --workdir=/opt \
    -v `pwd`:/opt \
    -v /tmp:/tmp \
    fabeschan/tfimg $@
