#!/bin/bash
##---------------------------------------------------------------------------##
## File  : regression/darwin-job-launch.sh
## Date  : Tuesday, May 31, 2016, 14:48 pm
## Author: Kelly Thompson
## Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
##         All rights are reserved.
##---------------------------------------------------------------------------##

# called from regression-master.sh
# assumes the following variables are defined in regression-master.sh:
#    $regdir     - /scratch/regress
#    $rscriptdir - /scratch/regress/draco/regression (actually, the location
#                  where the active regression_master.sh is located)
#    $subproj    - 'draco', 'jaynne', 'capsaicin', etc.
#    $build_type - 'Debug', 'Release'
#    $extra_params - '', 'intel13', 'pgi', 'coverage'

# command line arguments
args=( "$@" )
nargs=${#args[@]}
scriptname=${0##*/}
host=`uname -n`

#export MOABHOMEDIR=/opt/MOAB
export SHOWQ=/bin/squeue

# Dependencies: wait for these jobs to finish
dep_jobids=""
for (( i=0; i < $nargs ; ++i )); do
   dep_jobids="${dep_jobids} ${args[$i]} "
done

# sanity check
if [[ ! ${regdir} ]]; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'regdir' in the environment!"
    exit 1
fi
if [[ ! ${rscriptdir} ]]; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'rscriptdir' in the environment!"
    exit 1
fi
if [[ ! ${subproj} ]]; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'subproj' in the environment!"
    exit 1
fi
if [[ ! ${build_type} ]]; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'build_type' in the environment!"
    exit 1
fi
if [[ ! ${logdir} ]]; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'logdir' in the environment!"
    exit 1
fi

if test $subproj == draco || test $subproj == jayenne; then
  if [[ ! ${featurebranch} ]]; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'featurebranch' in the environment!"
    echo "printenv -> "
    printenv
  fi
fi

# Banner
echo "==========================================================================="
echo "Darwin Regression job launcher for ${subproj} - ${build_type} flavor."
echo "==========================================================================="
echo " "
echo "Environment:"
echo "   subproj        = ${subproj}"
echo "   build_type     = ${build_type}"
if [[ ! ${extra_params} ]]; then
  echo "   extra_params   = none"
else
  echo "   extra_params   = ${extra_params}"
fi
if [[ ${featurebranch} ]]; then
  echo "   featurebranch  = ${featurebranch}"
fi
echo "   regdir         = ${regdir}"
echo "   rscriptdir     = ${rscriptdir}"
echo "   logdir         = ${logdir}"
echo "   dashboard_type = ${dashboard_type}"
echo "   build_autodoc  = ${build_autodoc}"
echo " "
echo "   ${subproj}: dep_jobids = ${dep_jobids}"
echo " "

# Prerequisits:
# Wait for all dependencies to be met before creating a new job

for jobid in ${dep_jobids}; do
    while [ `ps --no-headers -u ${USER} -o pid | grep -c ${jobid}` -gt 0 ]; do
       echo "   ${subproj}: waiting for jobid = $jobid to finish (sleeping 5 minutes)."
       sleep 5m
    done
done

if ! test -d $logdir; then
  mkdir -p $logdir
  chgrp draco $logdir
  chmod g+rwX $logdir
  chmod g+s $logdir
fi

# Tell ctest to checkout jayenne from a local directory instead of gitlab. Also,
# update the local repository to include any feature branches that might be
# requested.
export gitroot=/usr/projects/draco/regress/git

darwin_regress_script="${rscriptdir}/darwin-regress.msub"

##---------------------------------------------------------------------------##
# Proposed: Init, Configure, Build, Test and Submit
# (1) Configure, build and test from the backend
# (2) Submit results from the front end.
# Submit from the front end

# Configure, Build, Test on back end
echo " "
echo "Configure, Build, Test:"
export REGRESSION_PHASE=cbt
cmd="/usr/bin/sbatch -v -o ${logdir}/darwin-${subproj}-${build_type}${epdash}${extra_params}${prdash}${featurebranch}-${REGRESSION_PHASE}.log -e ${regdir}/logs/darwin-${subproj}-${build_type}${epdash}${extra_params}${prdash}${featurebranch}-${REGRESSION_PHASE}.log ${darwin_regress_script}"
echo "${cmd}"
jobid=`eval ${cmd}`
# trim extra whitespace from number
jobid=`echo ${jobid//[^0-9]/}`

# Wait for CBT (Config, build, test) to finish
sleep 1m
while test "`$SHOWQ | grep $jobid`" != ""; do
   $SHOWQ | grep $jobid
   echo "   ${subproj}: waiting for jobid = $jobid to finish (sleeping 5 minutes)."
   sleep 5m
done

# Submit from the front end
echo " "
echo "Submit:"
export REGRESSION_PHASE=s
echo "Jobs done, now submitting ${build_type} results from darwin."
cmd="${darwin_regress_script} >& ${logdir}/darwin-${subproj}-${build_type}${epdash}${extra_params}${prdash}${featurebranch}-${REGRESSION_PHASE}.log"
echo "${cmd}"
eval "${cmd}"

echo "All done."

##---------------------------------------------------------------------------##
## End of darwin-job-launch.sh
##---------------------------------------------------------------------------##
