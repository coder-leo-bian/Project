#!/bin/sh
function_start()
{
  uwsgi -d --ini /root/Project/app/uwsgi.ini
}

function_stop()
{
  ps aux|grep uwsgi.ini | grep -v 'grep'|awk '{print $2}'| xargs kill -9
}

function_restart()
{
  function_dev_stop
  function_dev_start
}

if [ "$1" = "start" ] ; then
  function_start
elif [ "$1" = "stop" ] ; then
  function_stop
elif [ "$1" = "restart" ] ; then
  function_restart
else
    echo "Usage: $0 {start|stop}\n"
fi