#-----------------------------*-cmake-*----------------------------------------#
# file   draco/autodoc/CMakeLists.txt
# author Kelly Thompson <kgt@lanl.gov>
# date   2010 Oct 14
# brief  Instructions for building Doxygen documentation
# note   Copyright (C) 2010-2018 Los Alamos National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#

# 2018-03-10 (KT): As of cmake-3.9.0, we have the option of using the cmake
# function doxygen_add_docs() instead of the complex logic below.  We would
# lose a few features, but wouldn't need to maintain this file.
# Ref. https://cmake.org/cmake/help/v3.10/module/FindDoxygen.html

# config/vendor_libraries.cmake is responsible for detecting if doxygen is
# available.

if( NOT DOXYGEN_FOUND )
  return()
endif()

 # if AUTODOCDIR is set in environment (or make command line), create a CMake
 # variable with this value.
 if( ENV{AUTODOCDIR} )
    set( AUTODOCDIR "$ENV{AUTODOCDIR}" )
 endif()
 # if AUTODOCDIR is set, use it, otherwise, default to CMAKE_INSTALL_PREFIX.
 if( AUTODOCDIR )
    set( DOXYGEN_OUTPUT_DIR "${AUTODOCDIR}" )
  else()
    set( DOXYGEN_OUTPUT_DIR ${CMAKE_INSTALL_PREFIX}/autodoc )
 endif()

# Source directories that should be examined by doxygen Sets ${DOXYGEN_INPUT}.
# This value is embedded in the doxygen_config file below.
set( DOXYGEN_INPUT
   "${PROJECT_SOURCE_DIR}/autodoc ${PROJECT_BINARY_DIR}/autodoc" )

# BUG: Move this list generation into component_macros.cmake so that inactive 
#      packages are not included in this list.
file( GLOB package_list ${PROJECT_SOURCE_DIR}/src/* )
foreach( package ${package_list} )
   if( EXISTS ${package}/CMakeLists.txt )
      set( DOXYGEN_INPUT "${DOXYGEN_INPUT} ${package}" )
   endif()
   if( EXISTS ${package}/test/CMakeLists.txt )
      set( DOXYGEN_INPUT "${DOXYGEN_INPUT} ${package}/test" )
      set( DOXYGEN_EXAMPLES "${DOXYGEN_EXAMPLES} ${package}/test" )
   endif()
   if( EXISTS ${package}/autodoc )
      set( DOXYGEN_INPUT "${DOXYGEN_INPUT} ${package}/autodoc" )
   endif()
endforeach()

# Tell doxygen where image files are located so they can be copied to the 
# output directory.
#
# The list of source files (this variable also set by 
# comonent_macros.cmake::process_autodoc_pages()
list(APPEND DOXYGEN_IMAGE_PATH 
  "${PROJECT_SOURCE_DIR}/autodoc"
  "${PROJECT_SOURCE_DIR}/autodoc/html" )
list( REMOVE_DUPLICATES DOXYGEN_IMAGE_PATH )
# convert list of image directories into a space delimited string
unset( temp )
foreach( image_dir ${DOXYGEN_IMAGE_PATH} )
   set( temp "${temp} ${image_dir}" )
endforeach()
set( DOXYGEN_IMAGE_PATH "${temp}" )
unset( temp )

# Also look for files that have been configured (.in files) and
# placed in the BINARY_DIR.
file( GLOB package_list ${PROJECT_BINARY_DIR}/src/* )
foreach( package ${package_list} )
   if( EXISTS ${package}/autodoc )
      set( DOXYGEN_INPUT "${DOXYGEN_INPUT} ${package}/autodoc" )
   endif()
endforeach()

set( DOXYGEN_HTML_OUTPUT draco )
set( project_brief "Object-oriented components for numerically intensive, parallel, radiation transport applications.")
string( REPLACE " " "___" project_brief ${project_brief} )

# doxygen_config.in needs the following symbols:
add_custom_command(
  OUTPUT  "${PROJECT_BINARY_DIR}/autodoc/doxygen_config"
  COMMAND "${CMAKE_COMMAND}"
          -DINFILE="${PROJECT_SOURCE_DIR}/config/doxygen_config.in"
          -DOUTFILE="${PROJECT_BINARY_DIR}/autodoc/doxygen_config"
          -DPROJECT_NAME=${PROJECT_NAME}
          -Dpackage=${PROJECT_NAME}
          -DPROJECT_NUMBER="${Draco_VERSION_FULL}"
          -Dnumber="${Draco_VERSION_FULL}"
          -DOUTPUT_DIRECTORY="${DOXYGEN_OUTPUT_DIR}"
          -Ddoxygen_output_top="${DOXYGEN_OUTPUT_DIR}"
          -DINPUT=${DOXYGEN_INPUT}
          -Ddoxygen_input=${DOXYGEN_INPUT}
          -DEXAMPLE_PATH=${DOXYGEN_EXAMPLES}
          -Ddoxygen_examples=${DOXYGEN_EXAMPLES}
          -DHTML_OUTPUT=${DOXYGEN_HTML_OUTPUT}
          -Ddoxygen_html_output=${DOXYGEN_HTML_OUTPUT}
          -DDOTFILE_DIRS=""
          -Ddotpath=""
          -Dproject_brief=\"${project_brief}\"
          -Dlatex_yes_no=NO
          -Ddoxygen_latex_output=${DOXYGEN_HTML_OUTPUT}/../latex
          -Dhtmlheader="${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/header.html"
          -Dhtmlfooter="${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/footer.html"
          -Dproject_logo="${PROJECT_SOURCE_DIR}/autodoc/html/dragon.jpg"
          -Dnumthreads=${MPIEXEC_MAX_NUMPROCS}
          -DDOXYGEN_IMAGE_PATH=${DOXYGEN_IMAGE_PATH}
          -P "${PROJECT_SOURCE_DIR}/config/configureFileOnMake.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/config/doxygen_config.in" 
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/autodoc)
configure_file(
  "${PROJECT_SOURCE_DIR}/autodoc/html/footer.html.in"
  "${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/footer.html"
  @ONLY )
configure_file(
  "${PROJECT_SOURCE_DIR}/autodoc/html/header.html.in"
  "${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/header.html"
  @ONLY )
add_custom_command(
  OUTPUT  "${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/doxygen.css"
  COMMAND "${CMAKE_COMMAND}"
          -DINFILE="${PROJECT_SOURCE_DIR}/autodoc/html/doxygen.css"
          -DOUTFILE="${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/doxygen.css"
          -P "${PROJECT_SOURCE_DIR}/config/configureFileOnMake.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/autodoc/html/doxygen.css" )
add_custom_command(
  OUTPUT  "${PROJECT_BINARY_DIR}/autodoc/mainpage.dcc"
  COMMAND "${CMAKE_COMMAND}"
          -DINFILE="${PROJECT_SOURCE_DIR}/autodoc/mainpage.dcc.in"
          -DOUTFILE="${PROJECT_BINARY_DIR}/autodoc/mainpage.dcc"
          -DCOMP_LINKS=${COMP_LINKS}
          -DPACKAGE_LINKS=${PACKAGE_LINKS}
          -P "${PROJECT_SOURCE_DIR}/config/configureFileOnMake.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/autodoc/mainpage.dcc.in"
  COMMENT "Doxygen docs are installed at"
          "${DOXYGEN_HTML_OUTPUT}.  " )
add_custom_command(
  OUTPUT  "${PROJECT_BINARY_DIR}/autodoc/build_system.dcc"
  COMMAND "${CMAKE_COMMAND}"
          -DINFILE="${PROJECT_SOURCE_DIR}/autodoc/build_system.dcc.in"
          -DOUTFILE="${PROJECT_BINARY_DIR}/autodoc/build_system.dcc"
          -DCOMP_LINKS=${COMP_LINKS}
          -DPACKAGE_LINKS=${PACKAGE_LINKS}
          -P "${PROJECT_SOURCE_DIR}/config/configureFileOnMake.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/autodoc/build_system.dcc.in" )          

# The actual doxygen command that will be run
add_custom_command(
  OUTPUT
     ${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/index.html
     ${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/always_out_of_date.html
  COMMAND "${DOXYGEN_EXECUTABLE}" ${PROJECT_BINARY_DIR}/autodoc/doxygen_config
  DEPENDS
    ${PROJECT_BINARY_DIR}/autodoc/doxygen_config
    ${PROJECT_BINARY_DIR}/autodoc/mainpage.dcc
    ${PROJECT_BINARY_DIR}/autodoc/build_system.dcc
    ${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/doxygen.css
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/autodoc )

# Target for building doxygen files.
add_custom_target( autodoc
  DEPENDS ${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/always_out_of_date.html
  COMMENT "Building Doxygen based autodocs (HTML)...

*** Autodoc HTML pages are at
  file://${DOXYGEN_OUTPUT_DIR}/${DOXYGEN_HTML_OUTPUT}/index.html

" )

# To Do:
# 1. How to 'make clean'?
# 2. Consider using tagfiles and building each component as an thread
#    to decrease wall clock time needed to build docs.
