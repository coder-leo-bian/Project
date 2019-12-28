#!/bin/sh
function_start()
{
    python3 /root/Project/shell/mkdir_times_name.py
    ADDONS_PATH = `ls -td -- /opt/code/* | head -n 1`
    echo $ADDONS_PATH
    git clone https://github.com/bianjing2018/Project.git $ADDONS_PATH
    if [ -f "Project" ]; then
        unlink Project
    fi
    ln -s $ADDONS_PATH Project

}

if [ "$1" = "mk" ] ; then
  function_start
else
    echo "Usage: $0 {mk}\n"
fi

