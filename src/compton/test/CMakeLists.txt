#-----------------------------*-cmake-*----------------------------------------#
# file   compton/test/CMakeLists.txt
# author Kendra Keady
# date   2012 Aug 1
# brief  Generate build project files for compton/test.
# note   Copyright (C) 2016, Los Alamos National Security, LLC.
#        All rights reserved.

project( compton_test CXX )

# ---------------------------------------------------------------------------- #
# Source files
# ---------------------------------------------------------------------------- #

 set( test_sources
  ${PROJECT_SOURCE_DIR}/tCompton_NWA.cc )
 file( GLOB compton_data *.compton )

##---------------------------------------------------------------------------##
## Copy data files
##---------------------------------------------------------------------------##

foreach( file ${compton_data} )
  get_filename_component( shortname ${file} NAME )
  configure_file( ${file}
    ${PROJECT_BINARY_DIR}/${shortname} COPYONLY )
endforeach()

# ---------------------------------------------------------------------------- #
# Directories to search for include directives
# ---------------------------------------------------------------------------- #

include_directories(
  ${PROJECT_SOURCE_DIR}      # headers for tests
  ${PROJECT_BINARY_DIR}      # test/config.h
  ${PROJECT_SOURCE_DIR}/..   # headers for package
  )

# ---------------------------------------------------------------------------- #
# Build Unit tests
# ---------------------------------------------------------------------------- #

add_scalar_tests(
  SOURCES "${test_sources}"
  DEPS    "Lib_compton" )

provide_aux_files(
   FILES    "${compton_data}"
   SRC_EXT  ".compton"
   DEST_EXT ".compton" )
