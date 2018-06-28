#-----------------------------*-cmake-*----------------------------------------#
# file   mesh/test/CMakeLists.txt
# author Ryan Wollaeger <wollaeger@lanl.gov>
# date   Thursday, Jun 07, 2018, 15:44 pm
# brief  Generate build project files for mesh/test.
# note   Copyright (C) 2018, Los Alamos National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#
project( mesh_test CXX )

# ---------------------------------------------------------------------------- #
# Source files
# ---------------------------------------------------------------------------- #

# create unit test lists
set( one_pe_tests ${PROJECT_SOURCE_DIR}/tstDraco_Mesh.cc )
set( two_pe_tests ${PROJECT_SOURCE_DIR}/tstDraco_Mesh_DD.cc )

# ---------------------------------------------------------------------------- #
# Build Unit tests
# ---------------------------------------------------------------------------- #

add_parallel_tests(
    SOURCES "${one_pe_tests}"
    PE_LIST "1"
    DEPS    "Lib_c4;Lib_mesh" )

if( NOT ${DRACO_C4} STREQUAL "SCALAR" )
  add_parallel_tests(
    SOURCES "${two_pe_tests}"
    PE_LIST "2"
    DEPS    "Lib_c4;Lib_mesh" )
endif()

#------------------------------------------------------------------------------#
# End mesh/test/CMakeLists.txt
#------------------------------------------------------------------------------#
