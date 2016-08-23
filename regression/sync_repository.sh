#!/bin/bash
##---------------------------------------------------------------------------##
## File  : regression/sync_repository.sh
## Date  : Tuesday, May 31, 2016, 14:48 pm
## Author: Kelly Thompson
## Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
##         All rights are reserved.
##---------------------------------------------------------------------------##

# This script is used to mirror portions of the Jayenne and Capsaicin SVN
# repositories to a HPC location. The repository must be mirrored because the
# ctest regression system must be run from the HPC backend (via msub) where
# access to ccscs7:/ccs/codes/radtran/svn is not available.

#
# MODULES
#
# Determine if the module command is available
modcmd=`declare -f module`
# If not found, look for it in /usr/share/Modules (ML)
if [[ ! ${modcmd} ]]; then
   if test -f /usr/share/Modules/init/bash; then
      source /usr/share/Modules/init/bash
   else
      echo "ERROR: The module command was not found. No modules will be loaded."
   fi
fi
modcmd=`declare -f module`

#
# Environment
#

run () {
    echo $1
    if ! test $dry_run; then eval $1; fi
}

run "module load user_contrib svn git"

# Ensure that the permissions are correct
run "umask 0002"
svnhostmachine=ccscs7
MYHOSTNAME=`uname -n`
regdir=/usr/projects/jayenne/regress
svnroot=$regdir/svn
VENDOR_DIR=/usr/projects/draco/vendors
keychain=keychain-2.7.1

if ! test -d $regdir; then
  mkdir -p $regdir
fi

# Credentials via Keychain (SSH)
# http://www.cyberciti.biz/faq/ssh-passwordless-login-with-keychain-for-scripts
$VENDOR_DIR/$keychain/keychain $HOME/.ssh/cmake_dsa
if test -f $HOME/.keychain/$MYHOSTNAME-sh; then
    run "source $HOME/.keychain/$MYHOSTNAME-sh"
else
    echo "Error: could not find $HOME/.keychain/$MYHOSTNAME-sh"
fi

#
# update the scripts directories in /usr/projects/jayenne
#

if test -d $regdir/draco; then
  run "cd $regdir/draco; git pull"
else
  run "cd $regdir; git clone https://github.com/losalamos/Draco.git draco"
fi

if test -d $regdir/jayenne; then
  run "cd $regdir/jayenne; git pull"
else
  run "cd $regdir; git clone git@gitlab.lanl.gov:jayenne/jayenne.git"
fi

if test -d $regdir/capsaicin/scripts; then
    run "cd $regdir/capsaicin/scripts; svn update"
else
    run "mkdir -p $regdir/capsaicin; cd $regdir/capsaicin"
    run "svn co svn+ssh://$svnhostmachine/ccs/codes/radtran/svn/capsaicin/trunk/scripts"
fi

#
# Sync the repository to $svnroot
#
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

run "svnsync --non-interactive sync file:///${svnroot}/capsaicin"

#------------------------------------------------------------------------------#
# End sync_repository.sh
#------------------------------------------------------------------------------#
