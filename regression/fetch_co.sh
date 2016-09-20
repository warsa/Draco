#!/bin/bash

# fetch and checkout
# $1 is the path to git
# $2 is the host ('github' or 'gitlab')
# $3 is the pull request number (e.g.: 42)

GIT=$1
githost=$2
featurebranch=$3

if ! test -x $GIT; then
   echo "FATAL ERROR: unable to run GIT=$GIT."
   exit 1
fi
if [[ ! ${githost} ]]; then
   echo "FATAL ERROR: exected three arguments, example: fetch_co.sh <path to git> <github> <42>"
   echo "     second argument should be 'github' or 'gitlab'"
   exit 1
fi
if [[ ! ${featurebranch} ]]; then
   echo "FATAL ERROR: exected three arguments, example: fetch_co.sh <path to git> <github> <42>"
   echo "     third argument should be a bare integer (the pull request number)."
   exit 1
fi

if test $githost = "github"; then
  prdir="pull"
else
  prdir="merge-requests"
fi

cmd="${GIT} fetch origin ${prdir}/${featurebranch}/head:pr${featurebranch}"
echo $cmd
eval $cmd

cmd="${GIT} checkout pr${featurebranch}"
echo $cmd
eval $cmd
