docker run -it --rm \
    --privileged \
    --network host \
    --ipc=host \
    -v /home/unitree:/workspaces \
    -v /etc/localtime:/etc/localtime:ro \
    --runtime nvidia \
    tackgeun/gps-train:init \
    /bin/bash

