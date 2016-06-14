#-----------------------------*-cmake-*----------------------------------------#
# file   draco/regression/Draco_Win32.cmake
# author Kelly Thompson <kgt@lanl.gov>
# date   2016 June 14
# brief  CTest regression script for Draco on Win32.
# note   Copyright (C) 2016 Los Alamos National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#
# Ref: http://www.cmake.org/Wiki/CMake_Scripting_Of_CTest

cmake_minimum_required(VERSION 3.0.0)

# Use:
# - See draco/regression/regression_master.sh
# - Summary: The script must do something like this:
#   [export work_dir=/full/path/to/working/dir]
#   ctest [-V] [-VV] -S /path/to/this/script.cmake,\
#     [Experimental|Nightly|Continuous],\
#     [Debug[,Coverage]|Release|RelWithDebInfo]

set( CTEST_PROJECT_NAME "Draco" )
message("source ${CTEST_SCRIPT_DIRECTORY}/draco_regression_macros.cmake" )
include( "${CTEST_SCRIPT_DIRECTORY}/draco_regression_macros.cmake" )

# ====================================================================

set_defaults()
parse_args()
find_tools()
set_svn_command("draco/trunk")
# Add username and fully qualified machine name.
string( REPLACE "//ccscs7/" "//kellyt@ccscs7.lanl.gov/"
   CTEST_CVS_CHECKOUT ${CTEST_CVS_CHECKOUT} )
# Make machine name lower case
string( TOLOWER "${CTEST_SITE}" CTEST_SITE )

####################################################################
# The values in this section are optional you can either
# have them or leave them commented out
####################################################################

# this is the initial cache to use for the binary tree, be careful to escape
# any quotes inside of this string if you use it
set( CTEST_INITIAL_CACHE "
CMAKE_VERBOSE_MAKEFILE:BOOL=ON
CMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}
CMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
CTEST_CMAKE_GENERATOR:STRING=${CTEST_CMAKE_GENERATOR}
CTEST_USE_LAUNCHERS:STRING=${CTEST_USE_LAUNCHERS}
CTEST_TEST_TIMEOUT:STRING=${CTEST_TEST_TIMEOUT}

VENDOR_DIR:PATH=${VENDOR_DIR}
AUTODOCDIR:PATH=${AUTODOCDIR}

CMAKE_MAKE_PROGRAM:FILEPATH=${MAKECOMMAND}

${INIT_CACHE_PPE_PREFIX}
${TOOLCHAIN_SETUP}
# Set DRACO_DIAGNOSTICS and DRACO_TIMING:
${FULLDIAGNOSTICS}
")

message("CTEST_INITIAL_CACHE =
----------------------------------------------------------------------
${CTEST_INITIAL_CACHE}
----------------------------------------------------------------------")

message("
--> Draco_Linux64.cmake modes:
    CTEST_CONFIGURE = ${CTEST_CONFIGURE}
    CTEST_BUILD     = ${CTEST_BUILD}
    CTEST_TEST      = ${CTEST_TEST}
    CTEST_SUBMIT    = ${CTEST_SUBMIT}
")

message("Parsing ${CTEST_SOURCE_DIRECTORY}/CTestCustom.cmake")
ctest_read_custom_files("${CTEST_SOURCE_DIRECTORY}")

if( "${CTEST_CONFIGURE}" STREQUAL "ON" )
   # Empty the binary directory and recreate the CMakeCache.txt
   message( "ctest_empty_binary_directory( ${CTEST_BINARY_DIRECTORY} )" )
   ctest_empty_binary_directory( ${CTEST_BINARY_DIRECTORY} )
   # dummy command to give the file system time to catch up before creating CMakeCache.txt.
   file( WRITE d:/foo.txt ${CTEST_INITIAL_CACHE} )
   file( WRITE ${CTEST_BINARY_DIRECTORY}/CMakeCache.txt ${CTEST_INITIAL_CACHE} )
endif()

# Start
message( "ctest_start( ${CTEST_MODEL} )")
ctest_start( ${CTEST_MODEL} )

message( "${CTEST_COMMAND}" )

# Update and Configure
if( "${CTEST_CONFIGURE}" STREQUAL "ON" )
   message( "ctest_update( SOURCE ${CTEST_SOURCE_DIRECTORY} RETURN_VALUE res )"  )
   ctest_update( SOURCE ${CTEST_SOURCE_DIRECTORY} RETURN_VALUE res )
   message( "Files updated: ${res}" )

   message( "setup_for_code_coverage()" )
   setup_for_code_coverage() # from draco_regression_macros.cmake
   message(  "ctest_configure()" )
   ctest_configure(
      BUILD        "${CTEST_BINARY_DIRECTORY}"
      RETURN_VALUE res) # LABELS label1 [label2]
endif()

# Build
if( "${CTEST_BUILD}" STREQUAL "ON" )
   # Main build
   message( "ctest_build( TARGET install RETURN_VALUE res )" )
   ctest_build(
      TARGET install
      RETURN_VALUE res
      NUMBER_ERRORS num_errors
      NUMBER_WARNINGS num_warnings
      )
   message( "build result:
   ${res}
   Build errors  : ${num_errors}
   Build warnings: ${num_warnings}" )
endif()

# Test
if( "${CTEST_TEST}" STREQUAL "ON" )
   message( "ctest_test( PARALLEL_LEVEL ${MPIEXEC_MAX_NUMPROCS} SCHEDULE_RANDOM ON )" )
   ctest_test(
     PARALLEL_LEVEL  ${MPIEXEC_MAX_NUMPROCS}
     SCHEDULE_RANDOM ON
     TEST_LOAD       ${MPIEXEC_MAX_NUMPROCS} )
endif()

# Submit results
if( "${CTEST_SUBMIT}" STREQUAL "ON" )
   message( "ctest_submit()")
   ctest_submit()
endif()

message("end of ${CTEST_SCRIPT_NAME}.")

#------------------------------------------------------------------------------#
# End Draco_Win32.cmake
#------------------------------------------------------------------------------#
