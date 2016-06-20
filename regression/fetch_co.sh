#!/bin/bash

# fetch and checkout
# $1 is the path to git
# $2 is the pull request number (e.g.: 42)

GIT=$1
featurebranch=$2

if ! test -x $GIT; then
   echo "FATAL ERROR: unable to run GIT=$GIT."
   exit 1
fi
if test "${featurebranch}x" == "x"; then
   echo "FATAL ERROR: exected two arguments: fetch_co.sh <path to git> <42>"
   exit 1
fi

cmd="${GIT} fetch origin pull/${featurebranch}/head:pr${featurebranch}"
echo $cmd
eval $cmd

cmd="${GIT} checkout pr${featurebranch}"
echo $cmd
eval $cmd
