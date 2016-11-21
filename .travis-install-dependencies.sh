#!/bin/bash
##---------------------------------------------------------------------------##
## File  : ./travis-install-dependencies.sh
## Date  : Tuesday, Sep 20, 2016, 11:50 am
## Author: Kelly Thompson
## Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
##         All rights are reserved.
##---------------------------------------------------------------------------##

# Install tools and libraries that the Draco build system requires that are
# not provided by travis or by apt-get.

# preliminaries and environment
set -e
source regression/scripts/common.sh

topdir=`pwd` # /home/travis/build/losalamos/Draco
# HOME = /home/travis
# USER = travis
# GROUP = travis
RANDOM123_VER=1.09
CMAKE_VERSION=3.6.2-Linux-x86_64
NUMDIFF_VER=5.8.1
CLANG_FORMAT_VER=3.9
OPENMPI_VER=1.10.3

# printenv

if [[ ${STYLE} ]]; then

  # clang-format and git-clang-format
  echo " "
  echo "Clang-format"
  run "sudo add-apt-repository 'deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-${CLANG_FORMAT_VER} main'"
  run "wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key | sudo apt-key add -"
  run "sudo apt-get update -qq"
  run "sudo apt-get install -qq -y clang-format-${CLANG_FORMAT_VER}"
  run "cd ${VENDOR_DIR}/bin"
  run "ln -s /usr/bin/clang-format-${CLANG_FORMAT_VER} clang-format"
  run "ln -s /usr/bin/git-clang-format-${CLANG_FORMAT_VER} git-clang-format"
  run "cd $topdir"

else

  # Random123
  echo " "
  echo "Random123"
  cd $HOME
  run "wget https://www.deshawresearch.com/downloads/download_random123.cgi/Random123-${RANDOM123_VER}.tar.gz"
  run "tar -xvf Random123-${RANDOM123_VER}.tar.gz"
  echo "Please set RANDOM123_INC_DIR=$HOME/Random123-${RANDOM123_VER}/include"
  run "ls $HOME/Random123-${RANDOM123_VER}/include"

  # CMake
  echo " "
  echo "CMake"
  run "cd $HOME"
  run "wget --no-check-certificate http://www.cmake.org/files/v${CMAKE_VERSION:0:3}/cmake-${CMAKE_VERSION}.tar.gz"
  run "tar -xzf cmake-${CMAKE_VERSION}.tar.gz"
  run "cd $topdir"

  # Numdiff
  echo " "
  echo "Numdiff"
  run "wget http://mirror.lihnidos.org/GNU/savannah/numdiff/numdiff-${NUMDIFF_VER}.tar.gz"
  run "tar -xvf numdiff-${NUMDIFF_VER}.tar.gz"
  run "cd numdiff-${NUMDIFF_VER}"
  run "./configure --prefix=/usr && make"
  run "sudo make install"
  run "cd $topdir"

  # OpenMPI
  echo " "
  echo "OpenMPI"
  run "wget --no-check-certificate https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VER}.tar.gz"
  run "tar -zxf openmpi-${OPENMPI_VER}.tar.gz"
  run "cd openmpi-${OPENMPI_VER}"
  run "./configure --enable-mpi-thread-multiple --quiet > build-openmpi.log"
  run "make >> build-openmpi.log"
  run "sudo make install"
  run "sudo sh -c 'echo \"/usr/local/lib\n/usr/local/lib/openmpi\" > /etc/ld.so.conf.d/openmpi.conf'"
  run "sudo ldconfig"
  run "cd $topdir"

fi

# Finish up and report
