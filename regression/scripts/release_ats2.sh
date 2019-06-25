#!/bin/bash -l

##---------------------------------------------------------------------------##
## Assumptions:
##---------------------------------------------------------------------------##
## 1. Directory layout:
##    /usr/gapps/jayenne/draco-NN_NN_NN/
##                  scripts/release_ats2.sh  # this script
##                  logs/                    # build/test logs
##                  source/                  # git checkout of release branch
##                  flavor/opt|debug         # released libraries/headers
## 2. Assumes that this script lives at the location above when
##    executed.

##---------------------------------------------------------------------------##
## Instructions
##---------------------------------------------------------------------------##
## 1. Set modulefiles to be loaded in named environment functions.
## 2. Update variables that control the build:
##    - $ddir
## 3. Run this script: ./release_ats2.sh &> ../logs/release_rzansel.log

#----------------------------------------------------------------------#
# Per release settings go here (edits go here)
#----------------------------------------------------------------------#

# Draco install directory name (/usr/gapps/jayenne/draco-NN_NN_NN)
export package=draco
ddir=draco-7_2_0
pdir=$ddir

# release for each environment listed
# These are defined in cts-env.sh
environments="gcc731env"

# ============================================================================
# ====== Normally, you do not edit anything below this line ==================
# ============================================================================

##---------------------------------------------------------------------------##
## Generic setup (do not edit)
##---------------------------------------------------------------------------##

export script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export draco_script_dir=`readlink -f $script_dir`
echo "source $draco_script_dir/common.sh"
source ${draco_script_dir}/common.sh

# CMake options that will be included in the configuration step
CONFIG_BASE="-DDRACO_VERSION_PATCH=`echo $ddir | sed -e 's/.*_//'`"
CONFIG_BASE+=" -DCMAKE_VERBOSE_MAKEFILE=ON"
export CONFIG_BASE

# sets umask 0002
# sets $install_group, $install_permissions, $build_permissions
establish_permissions

export source_prefix="/usr/gapps/jayenne/$pdir"
(cd /usr/gapps/jayenne/; if [[ -d latest ]]; then rm latest; fi; ln -s $pdir latest)
scratchdir=`selectscratchdir`

# ppn=`showstats -n | tail -n 1 | awk '{print $3}'`
#build_pe=`npes_build`
#test_pe=`npes_test`

# SLURM
#avail_queues=`sacctmgr -np list assoc user=$LOGNAME | sed -e 's/.*|\(.*dev.*\|.*access.*\)|.*/\1/' | sed -e 's/|.*//'`

# case $avail_queues in
#   *access*) access_queue="-A access --qos=access" ;;
#   *dev*) access_queue="--qos=dev" ;;
# esac

# Make sure there is enough tmp space for the compiler's temporary files.
# export TMPDIR=/$scratchdir/$USER/tmp
# if ! test -d $TMPDIR; then
#   mkdir -p $TMPDIR
# fi
# if ! test -d $TMPDIR; then
#   echo "Could not create TMPDIR=$TMPDIR."
#   exit 1
# fi

# =============================================================================
# Build types:
# - These must be copied into release_toss.msub because bash arrays cannot
#   be passed to the subshell (bash bug)
# =============================================================================

OPTIMIZE_ON="-DCMAKE_BUILD_TYPE=Release"
OPTIMIZE_OFF="-DCMAKE_BUILD_TYPE=Debug"
OPTIMIZE_RWDI="-DCMAKE_BUILD_TYPE=RELWITHDEBINFO -DDRACO_DBC_LEVEL=15"

LOGGING_ON="-DDRACO_DIAGNOSTICS=7 -DDRACO_TIMING=1"
LOGGING_OFF="-DDRACO_DIAGNOSTICS=0 -DDRACO_TIMING=0"

# Define the meanings of the various code versions:

VERSIONS=( "debug" "opt" "rwdi" )
OPTIONS=( "${OPTIMIZE_OFF} ${LOGGING_OFF}" "${OPTIMIZE_ON} ${LOGGING_OFF}" "${OPTIMIZE_RWDI} ${LOGGING_OFF}" )

##---------------------------------------------------------------------------##
## Environment review
##---------------------------------------------------------------------------##

verbose=1
if test $verbose == 1; then
  echo
  echo "Build environment summary:"
  echo "=========================="
  echo "script_dir     = $script_dir"
  echo "source_prefix  = $source_prefix"
  echo "log_dir        = $source_prefix/logs"
  echo "scratchdir     = $scratchdir/$USER"
  echo
  echo "package     = $package"
  echo "versions:"
  for (( i=0 ; i < ${#VERSIONS[@]} ; ++i )); do
    echo -e "   ${VERSIONS[$i]}, \t options = ${OPTIONS[$i]}"
  done
  echo
fi

##---------------------------------------------------------------------------##
## Execute the build, test and install
##---------------------------------------------------------------------------##

echo "source ${draco_script_dir}/ats2-env.sh"
source ${draco_script_dir}/ats2-env.sh

jobids=""
for env in $environments; do

  # Run the bash function defined above to load appropriate module
  # environment.
  echo -e "\nEstablish environment $env"
  echo "======================================="
  $env

  buildflavor=`flavor`
  # e.g.: buildflavor=snow-openmpi-1.6.5-intel-15.0.3

  export install_prefix="$source_prefix/$buildflavor"
  export build_prefix="$scratchdir/$USER/$pdir/$buildflavor"
  export draco_prefix="/usr/gapps/jayenne/$ddir/$buildflavor"

  for (( i=0 ; i < ${#VERSIONS[@]} ; ++i )); do

    export version=${VERSIONS[$i]}
    export options=${OPTIONS[$i]}

    export CONFIG_EXTRA="$CONFIG_BASE"

    # export dry_run=1
    export steps="config build test"
#    cmd="bsub -qpdebug -J release_draco $access_queue -W 1:00 -nnodes 1 \
#-eo $source_prefix/logs/release-$buildflavor-$version.log \
#$script_dir/release.msub"
    cmd="bsub -q pdebug -nnodes 1 -W 2:00 -J release_draco \
-o $source_prefix/logs/release-$buildflavor-$version.log \
-e $source_prefix/logs/release-$buildflavor-$version.log"
    echo -e "\nConfigure, Build and Test $buildflavor-$version version of $package.\n"
    echo "$cmd $script_dir/release.msub"
#    jobid=`eval ${cmd} &`
    jobid=`$cmd < $script_dir/release.msub`
    sleep 1m
    # trim extra whitespace from number
    jobid=`echo ${jobid//[^0-9]/}`
    export jobids="$jobid $jobids"

  done
done

##---------------------------------------------------------------------------##
## Set permissions
##---------------------------------------------------------------------------##

publish_release

##---------------------------------------------------------------------------##
## End
##---------------------------------------------------------------------------##
