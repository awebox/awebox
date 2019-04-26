#!/bin/bash
# http://stackoverflow.com/questions/151677/tool-for-adding-license-headers-to-source-files
for i in $(find awebox -name '*.py') # or whatever other pattern...
do
  if ! grep -q MERCHANTABILITY $i
  then
    cat COPYING $i >$i.new && mv $i.new $i
    echo $i
  fi
done