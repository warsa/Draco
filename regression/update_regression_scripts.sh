#!/bin/bash
##---------------------------------------------------------------------------##
## File  : regression/update_regression_scripts.sh
## Date  : Tuesday, May 31, 2016, 14:48 pm
## Author: Kelly Thompson
## Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
##         All rights are reserved.
##---------------------------------------------------------------------------##

umask 0002
target="`uname -n | sed -e s/[.].*//`"
MYHOSTNAME="`uname -n`"

# Locate the directory that this script is located in:
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# import some bash functions
source $scriptdir/scripts/common.sh

# Ensure that the permissions are correct
case ${target} in
  darwin-login*)
    echo "Please run regressions from darwin-fe instead of darwin-login."
    exit 1
    ;;
  darwin-fe* | cn[0-9]*)
    # personal copy of ssh-agent.
    export PATH=$HOME/bin:$PATH
    /usr/projects/draco/vendors/keychain-2.7.1/keychain $HOME/.ssh/cmake_dsa
    if test -f $HOME/.keychain/$MYHOSTNAME-sh; then
       source $HOME/.keychain/$MYHOSTNAME-sh
    fi

    # Load keytab: (see notes at draco/regression/push_repositories_xf.sh)
    # Use a different cache location to avoid destroying any active user's
    # kerberos.
    # export KRB5CCNAME=/tmp/regress_kerb_cache
    # Obtain kerberos authentication via keytab
    # run "kinit -l 1h -kt $HOME/.ssh/xfkeytab transfer/${USER}push@lanl.gov"

    #module unload subversion
    #module load subversion
    if test -d /projects/opt/centos7/subversion/1.9.2/bin; then
      export PATH=/projects/opt/centos7/subversion/1.9.2/bin:$PATH
    fi
    SVN=`which svn`
    # SVN=/projects/opt/centos7/subversion/1.9.2/bin/svnsync
    REGDIR=/usr/projects/draco/regress

    svnroot=/usr/projects/draco/regress/svn
    gitroot=/usr/projects/draco/regress/git
    if ! test -d; then
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
      #echo "at svn+ssh://ccscs7/ccs/codes/radtran/svn/draco."
      #exit 1
      #fi
      #exit 0
      # cp pre-revprop-change.tmpl pre-revprop-change; chmod 775 \
      #    pre-revprop-change
      # vi pre-revprop-change --> comment out all code.
      # cd $svnroot
      # svnsync init file:///${svnroot}/jayenne svn+ssh://ccscs7/ccs/codes/radtran/svn/jayenne
      # svnsync sync file:///${svnroot}/jayenne
    fi
    # if ! test -d $gitroot; then
    #   export https_proxy=http://proxyout.lanl.gov:8080
    #   export HTTPS_PROXY=$https_proxy
    #   run "mkdir -p $gitroot"
    #   (run "cd $gitroot; git clone https://github.com/losalamos/Draco.git draco")
    # fi

    run "${SVN}sync --non-interactive sync file://${svnroot}/jayenne"
    run "${SVN}sync --non-interactive sync file://${svnroot}/capsaicin"
    # (run "cd $gitroot/draco; git pull origin develop")
    ;;
  ccscs*)
    REGDIR=/scratch/regress
    SVN=/scratch/vendors/subversion-1.9.3/bin/svn
    /scratch/vendors/keychain-2.8.2/keychain $HOME/.ssh/cmake_dsa
    if test -f $HOME/.keychain/$MYHOSTNAME-sh; then
       source $HOME/.keychain/$MYHOSTNAME-sh
    fi
    ;;
  *)
    # module load user_contrib subversion
    SVN=/scratch/vendors/subversion-1.9.3/bin/svn
    REGDIR=/scratch/regress
    ;;
esac

# Update main regression scripts
run "cd ${REGDIR}/draco; git pull"
run "cd ${REGDIR}/jayenne/regression; ${SVN} update"
run "cd ${REGDIR}/capsaicin/scripts; ${SVN} update"

##---------------------------------------------------------------------------##
## End update_regression_scripts.sh
##---------------------------------------------------------------------------##
