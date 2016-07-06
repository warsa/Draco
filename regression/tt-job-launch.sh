#!/bin/bash
##---------------------------------------------------------------------------##
## File  : regression/tt-job-launch.sh
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
scriptname=`basename $0`
host=`uname -n`

export SHOWQ=/opt/MOAB/bin/showq

 # Dependencies: wait for these jobs to finish
dep_jobids=""
for (( i=0; i < $nargs ; ++i )); do
   dep_jobids="${dep_jobids} ${args[$i]} "
done

# sanity check
if test "${regdir}x" = "x"; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'regdir' in the environment!"
    exit 1
fi
if test "${rscriptdir}x" = "x"; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'rscriptdir' in the environment!"
    exit 1
fi
if test "${subproj}x" = "x"; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'subproj' in the environment!"
    exit 1
fi
if test "${build_type}x" = "x"; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'build_type' in the environment!"
    exit 1
fi
if test "${logdir}x" = "x"; then
    echo "FATAL ERROR in ${scriptname}: You did not set 'logdir' in the environment!"
    exit 1
fi

# Banner
echo "==========================================================================="
echo "Trinitite Regression job launcher for ${subproj} - ${build_type} flavor."
echo "==========================================================================="
echo " "
echo "Environment:"
echo "   subproj        = ${subproj}"
echo "   build_type     = ${build_type}"
if test "${extra_params}x" == "x"; then
echo "   extra_params   = none"
else
echo "   extra_params   = ${extra_params}"
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
    while [ `ps --no-headers -u ${USER} -o pid | grep ${jobid} | wc -l` -gt 0 ]; do
       echo "   ${subproj}: waiting for jobid = $jobid to finish (sleeping 5 min)."
       sleep 5m
    done
done

# Configure, Build on front end
echo "Configure, Build:"
export REGRESSION_PHASE=cb
echo " "
echo "Configure and Build on the front end..."
cmd="${rscriptdir}/tt-regress.msub >& ${logdir}/${machine_name_short}-${subproj}-${build_type}${epdash}${extra_params}${prdash}${featurebranch}-${REGRESSION_PHASE}.log"
echo "${cmd}"
eval "${cmd}"

# Wait for CB (Configure and Build) before starting the testing and
# reporting from the login node:
echo " "
echo "Test:"
export REGRESSION_PHASE=t
echo " "
echo "Test from the login node..."
cmd="/opt/MOAB/bin/msub -j oe -V -o ${logdir}/${machine_name_short}-${subproj}-${build_type}${epdash}${extra_params}${prdash}${featurebranch}-${REGRESSION_PHASE}.log ${rscriptdir}/tt-regress.msub"
echo "${cmd}"
jobid=`eval ${cmd}`
jobid=`echo $jobid | sed '/^$/d'`
echo "jobid = ${jobid}"

# Wait for testing to finish
sleep 1m
while test "`${SHOWQ} | grep $jobid`" != ""; do
   ${SHOWQ} | grep $jobid
   sleep 1m
done

# Submit from the front end
echo " "
echo "Submit:"
export REGRESSION_PHASE=s
echo "Jobs done, now submitting ${build_type} results from tt-fey."
cmd="${rscriptdir}/tt-regress.msub >& ${logdir}/${machine_name_short}-${subproj}-${build_type}${epdash}${extra_params}${prdash}${featurebranch}-${REGRESSION_PHASE}.log"
echo "${cmd}"
eval "${cmd}"

# Submit from the front end
echo "Jobs done."

##---------------------------------------------------------------------------##
## End of tt-job-launch.sh
##---------------------------------------------------------------------------##
