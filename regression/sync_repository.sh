#!/bin/bash
##---------------------------------------------------------------------------##
## File  : regression/sync_repository.sh
## Date  : Tuesday, May 31, 2016, 14:48 pm
## Author: Kelly Thompson
## Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
##         All rights are reserved.
##---------------------------------------------------------------------------##

# This script is used for 2 similar but distinct operations:
#
# 1. It mirrors portions of the Capsaicin SVN repositories to a HPC
#    location. The repository must be mirrored because the ctest regression
#    system must be run from the HPC backend (via msub) where access to
#    ccscs7:/ccs/codes/radtran/svn is not available.
# 2. It also mirrors git@github.com/losalamos/Draco.git and
#    git@gitlab.lanl.gov/jayenne/jayenne.git to these locations:
#    - ccscs7:/ccs/codes/radtran/git
#    - darwin-fe:/usr/projects/draco/regress/git
#    On ccscs7, this is done to allow Redmine to parse the current repository
#    preseting a GUI view and scraping commit information that connects to
#    tracked issues. On darwin, this is done to allow the regressions running on
#    the compute node to access the latest git repository. This also copies down
#    all pull requests.

target="`uname -n | sed -e s/[.].*//`"

# Locate the directory that this script is located in:
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# MODULES
#
# Determine if the module command is available
modcmd=`declare -f module`
# If not found, look for it in /usr/share/Modules (ML)
if [[ ! ${modcmd} ]]; then
  case ${target} in
    tt-fey*) module_init_dir=/opt/cray/pe/modules/3.2.10.4/init ;;
    # snow (Toss3)
    sn-fey*) module_init_dir=/usr/share/lmod/lmod/init ;;
    # ccs-net, darwin, ml
    *)       module_init_dir=/usr/share/Modules/init ;;
  esac
  if test -f ${module_init_dir}/bash; then
    source ${module_init_dir}/bash
  else
    echo "ERROR: The module command was not found. No modules will be loaded."
  fi
  modcmd=`declare -f module`
fi

#
# Environment
#

# import some bash functions
source $scriptdir/scripts/common.sh

# Ensure that the permissions are correct
run "umask 0002"
svnhostmachine=ccscs7

case ${target} in
  ccscs*)
    run "module load user_contrib subversion git"
    regdir=/scratch/regress
    gitroot=/ccs/codes/radtran/git
    VENDOR_DIR=/scratch/vendors
    keychain=keychain-2.8.2
    ;;
  darwin-fe* | cn[0-9]*)
    regdir=/usr/projects/draco/regress
    gitroot=$regdir/git
    svnroot=$regdir/svn
    VENDOR_DIR=/usr/projects/draco/vendors
    keychain=keychain-2.7.1
    ;;
  ml-fey*)
    run "module load user_contrib subversion git"
    regdir=/usr/projects/jayenne/regress
    gitroot=$regdir/git
    svnroot=$regdir/svn
    VENDOR_DIR=/usr/projects/draco/vendors
    keychain=keychain-2.7.1
    ;;
  sn-fey*)
    run "module load user_contrib subversion git"
    regdir=/usr/projects/jayenne/regress
    gitroot=$regdir/git.sn
    VENDOR_DIR=/usr/projects/draco/vendors
    keychain=keychain-2.7.1
    ;;
  tt-fey*)
    run "module use /usr/projects/hpcsoft/cle6.0/modulefiles/trinitite/misc"
    run "module use /usr/projects/hpcsoft/cle6.0/modulefiles/trinitite/tools"
    run "module load user_contrib subversion git"
    regdir=/usr/projects/jayenne/regress
    gitroot=$regdir/git.tt
    VENDOR_DIR=/usr/projects/draco/vendors
    keychain=keychain-2.7.1
    ;;
esac

if ! test -d $regdir; then
  mkdir -p $regdir
fi

# Credentials via Keychain (SSH)
# http://www.cyberciti.biz/faq/ssh-passwordless-login-with-keychain-for-scripts
MYHOSTNAME="`uname -n`"
$VENDOR_DIR/$keychain/keychain $HOME/.ssh/cmake_dsa
if test -f $HOME/.keychain/$MYHOSTNAME-sh; then
  run "source $HOME/.keychain/$MYHOSTNAME-sh"
else
  echo "Error: could not find $HOME/.keychain/$MYHOSTNAME-sh"
fi

# ---------------------------------------------------------------------------- #
# Create copies of SVN and GIT repositories on the local file system
#
# Locations:
# - ccs-net servers:
#   /ccs/codes/radtran/git
# - HPC (moonlight, snow, trinitite, etc.)
#   /usr/projects/draco/jayenne/regress/[git|svn]
#
# Keep local copies of the github, gitlab and svn repositories. This local
# filesystem location is needed for our regression system on some systems where
# the back-end worker nodes can't see the outside world. Additionally, the
# output produced by downloading the repository to the local filesystem can be
# parsed so that CI regressions can be started. On the CCS-NET, the local copy
# is also visible to Redmine so changes in the repository will show up in the
# datase (GUI repository, wiki/ticket references to commits).
# ---------------------------------------------------------------------------- #

# DRACO: For all machines running this scirpt, copy all of the git repositories
# to the local file system.

# Store some output into a local file to simplify parsing.
TMPFILE_DRACO=$(mktemp /var/tmp/draco_repo_sync.XXXXXXXXXX) || { echo "Failed to create temporary file"; exit 1; }

echo " "
echo "Copy Draco git repository to the local file system..."
if test -d $gitroot/Draco.git; then
  run "cd $gitroot/Draco.git"
  run "git fetch origin +refs/heads/*:refs/heads/*"
  run "git fetch origin +refs/pull/*:refs/pull/*" &> $TMPFILE_DRACO
  cat $TMPFILE_DRACO
  run "git reset --soft"
else
  run "mkdir -p $gitroot"
  run "cd $gitroot"
  run "git clone --bare git@github.com:losalamos/Draco.git Draco.git"
fi

# JAYENNE: For all machines running this scirpt, copy all of the git repositories
# to the local file system.

# Store some output into a local file to simplify parsing.
TMPFILE_JAYENNE=$(mktemp /var/tmp/jayenne_repo_sync.XXXXXXXXXX) || { echo "Failed to create temporary file"; exit 1; }
echo " "
echo "Copy Jayenne git repository to the local file system..."
if test -d $gitroot/jayenne.git; then
  run "cd $gitroot/jayenne.git"
  run "git fetch origin +refs/heads/*:refs/heads/*"
  run "git fetch origin +refs/merge-requests/*:refs/merge-requests/*" &> $TMPFILE_JAYENNE
  cat $TMPFILE_JAYENNE
  run "git reset --soft"
else
  run "mkdir -p $gitroot; cd $gitroot"
  run "git clone --bare git@gitlab.lanl.gov:jayenne/jayenne.git jayenne.git"
fi
case ${target} in
  ccscs7*)
    # Keep a copy of the bare repo for Redmine.  This version doesn't have the
    # PRs since this seems to confuse Redmine.
    echo " "
    echo "(Redmine) Copy Draco git repository to the local file system..."
    if test -d $gitroot/Draco.git; then
      run "cd $gitroot/Draco-redmine.git"
      run "git fetch origin +refs/heads/*:refs/heads/*"
      run "git reset --soft"
    fi
    echo " "
    echo "(Redmine) Copy Jayenne git repository to the local file system..."
    if test -d $gitroot/jayenne.git; then
      run "cd $gitroot/jayenne-redmine.git"
      run "git fetch origin +refs/heads/*:refs/heads/*"
      run "git reset --soft"
    fi
    ;;
  ml-fey* | darwin-fe* )
    # CAPSAICIN: svn-sync the repository to the local file system.
    if ! test -d $svnroot; then
      echo "*** ERROR ***"
      echo "*** SVN repository not found ***"
      exit 1
      # http://journal.paul.querna.org/articles/2006/09/14/using-svnsync/
      # mkdir -p ${svnroot}; cd ${svnroot}
      # svnadmin create ${svnroot}/jayenne
      # chgrp -R draco jayenne; chmod -R g+rwX,o=g-w jayenne
      # cd jayenne/hooks
      # cp pre-commit.tmpl pre-commit; chmod 775 pre-commit
      # vi pre-commit; comment out all code and add...
      #if ! test `whoami` = 'kellyt'; then
      #echo "This is a read only repository.  The real SVN repository is"
      #echo "at svn+ssh://ccscs8/ccs/codes/radtran/svn/draco."
      #exit 1
      #fi
      #exit 0
      # cp pre-revprop-change.tmpl pre-revprop-change; chmod 775 \
      #    pre-revprop-change
      # vi pre-revprop-change --> comment out all code.
      # cd $svnroot
      # svnsync init file:///${svnroot}/jayenne svn+ssh://ccscs8/ccs/codes/radtran/svn/jayenne
      # svnsync sync file:///${svnroot}/jayenne
    fi

    echo " "
    echo "Copy capsaicin svn repository to the local file system..."
    run "svnsync --non-interactive sync file:///${svnroot}/capsaicin"
    ;;
esac

#------------------------------------------------------------------------------#
# Continuous Integration Hooks:
#
# Extract a list of PRs that are new and optionally start regression run by
# parsing the output of 'git fetch' from above.
#
# The output from the above command may include text of the form:
#   [new ref]        refs/pull/84/head -> refs/pull/84/head <-- new PR
#   03392b8..fd3eabc refs/pull/86/head -> refs/pull/86/head <-- updated PR
#                    Extract a list of PRs that are new and optionally start
#                    regression run
# ------------------------------------------------------------------------------#

# Draco CI ------------------------------------------------------------
draco_prs=`grep 'refs/pull/[0-9]*/head$' $TMPFILE_DRACO | awk '{print $NF}'`
for prline in $draco_prs; do
  case ${target} in

    # CCS-NET: Coverage (Debug) & Valgrind (Debug)
    ccscs*)
      # Coverage (Debug) & Valgrind (Debug)
      pr=`echo $prline |  sed -r 's/^[^0-9]*([0-9]+).*/\1/'`
      logfile=$regdir/logs/ccscs-draco-Debug-coverage-master-pr${pr}.log
      echo "- Starting regression (coverage) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Debug -e coverage \
        -p draco -f pr${pr} &> $logfile &
      logfile=$regdir/logs/ccscs-draco-Debug-valgrind-master-pr${pr}.log
      echo "- Starting regression (valgrind) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Debug -e valgrind \
        -p draco -f pr${pr} &> $logfile &
      ;;

    # Moonlight: Fulldiagnostics (Debug)
    ml-fey*)
      pr=`echo $prline |  sed -r 's/^[^0-9]*([0-9]+).*/\1/'`
      logfile=$regdir/logs/ml-draco-Debug-fulldiagnostics-master-pr${pr}.log
      echo "- Starting regression (fulldiagnostics) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Debug \
        -e fulldiagnostics -p draco -f pr${pr} &> $logfile &
      ;;

    # Snow ----------------------------------------
    sn-fe*)
      # No CI
      ;;

    # Trinitite: Release
    tt-fey*)
      pr=`echo $prline |  sed -r 's/^[^0-9]*([0-9]+).*/\1/'`
      logfile=$regdir/logs/tt-draco-Release-master-pr${pr}.log
      echo "- Starting regression (Release) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Release -p draco \
        -f pr${pr} &> $logfile &
      ;;

    # Darwin ----------------------------------------
    darwin-fe*)
      # No CI
      ;;
  esac
done

# Jayenne CI ------------------------------------------------------------
jayenne_prs=`grep 'refs/merge-requests/[0-9]*/head$' $TMPFILE_JAYENNE | awk '{print $NF}'`
ipr=0 # count the number of PRs processed. Only the first needs to build draco.
for prline in $jayenne_prs; do

  pr=`echo $prline |  sed -r 's/^[^0-9]*([0-9]+).*/\1/'`

  if [[ $ipr == 0 ]]; then
    projects="draco jayenne"
    featurebranches="develop pr${pr}"
  else
    projects="jayenne"
    featurebranches="pr${pr}"
  fi
  ((ipr++))

  case ${target} in

    # CCS-NET: Coverage (Debug) & Valgrind (Debug)
    ccscs*)
      logfile=$regdir/logs/ccscs-jayenne-Debug-coverage-master-pr${pr}.log
      echo "- Starting regression (coverage) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Debug -e coverage \
        -p "${projects}" -f "${featurebranches}" &> $logfile &

      logfile=$regdir/logs/ccscs-jayenne-Debug-valgrind-master-pr${pr}.log
      echo "- Starting regression (valgrind) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Debug -e valgrind \
        -p "${projects}" -f "${featurebranches}" &> $logfile &
      ;;

    # Moonlight: Fulldiagnostics (Debug)
    ml-fey*)
      logfile=$regdir/logs/ml-jayenne-Debug-fulldiagnostics-master-pr${pr}.log
      echo "- Starting regression (fulldiagnostics) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Debug \
        -e fulldiagnostics -p "${projects}" -f "${featurebranches}" \
        &> $logfile &
      ;;

    # Snow ----------------------------------------
    sn-fe*)
      # No CI
      ;;

    # Trinitite: Release
    tt-fey*)
      logfile=$regdir/logs/tt-jayenne-Release-master-pr${pr}.log
      echo "- Starting regression (Release) for pr${pr}."
      echo "  Log: $logfile"
      $regdir/draco/regression/regression-master.sh -r -b Release \
        -p "${projects}" -f "${featurebranches}" &> $logfile &
      ;;

    # Darwin ----------------------------------------
    darwin-fe*)
      # No CI
      ;;
  esac
done

#------------------------------------------------------------------------------#
# End sync_repository.sh
#------------------------------------------------------------------------------#
