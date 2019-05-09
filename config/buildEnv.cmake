#-----------------------------*-cmake-*----------------------------------------#
# file   buildEnv.cmake
# author Kelly Thompson <kgt@lanl.gov>
# date   2010 June 5
# brief  Default CMake build parameters
# note   Copyright (C) 2016-2019 Triad National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#

include_guard(GLOBAL)

#------------------------------------------------------------------------------#
# Build Parameters
#------------------------------------------------------------------------------#
macro( dbsSetDefaults )

  # make install less verbose.
  set(CMAKE_INSTALL_MESSAGE LAZY)

  # if undefined, force build_type to "release"
  if( NOT CMAKE_CONFIGURATION_TYPES )
     if( "${CMAKE_BUILD_TYPE}x" STREQUAL "x" )
        set( CMAKE_BUILD_TYPE "Debug" CACHE STRING "Release, Debug, RelWithDebInfo" FORCE )
     endif()
     # constrain pull down values in cmake-gui
     set_property( CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS Release Debug MinSizeRel RelWithDebInfo )
  endif()

  # Provide default value for install_prefix
  if( "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local" OR
      "${CMAKE_INSTALL_PREFIX}" MATCHES "C:/Program Files" )
     set( CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../install" )
     get_filename_component( CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" ABSOLUTE )
     set( CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH
        "Install path prefix, prepended onto install directories" FORCE)
  endif()
  mark_as_advanced( EXECUTABLE_OUTPUT_PATH )
  mark_as_advanced( LIBRARY_OUTPUT_PATH )
  mark_as_advanced( DART_TESTING_TIMEOUT )

  # For win32 platforms avoid copying all dependent dll libraries into the test
  # directories by using a common runtime directory.
  if( WIN32 )
     if( CMAKE_CONFIGURATION_TYPES )
        set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
     else() # nmake or mingw32-make
        set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE} )
     endif()
  endif()

  # Option for solution folders for GUI-based development environments
  if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
     option( BUILD_USE_SOLUTION_FOLDERS "Enable grouping of projects in VS" ON )
     set_property( GLOBAL PROPERTY USE_FOLDERS ${BUILD_USE_SOLUTION_FOLDERS} )
  endif()

  # Special setup for Visual Studio builds.
  if(WIN32 AND NOT UNIX AND NOT BORLAND AND NOT MINGW )
     set( CMAKE_SUPPRESS_REGENERATION ON )
  endif()

  if( CMAKE_CONFIGURATION_TYPES )
    set( Draco_BUILD_TYPE "Multi-config")
  else()
    string( TOUPPER ${CMAKE_BUILD_TYPE} Draco_BUILD_TYPE )
  endif()

  # Design-by-Contract
  if( NOT DEFINED DRACO_DBC_LEVEL )

    # Default is on (7), except for Makefile based Release builds .
    #   Insist() assertions    : always on
    #   Require() preconditions: add +1 to DBC_LEVEL
    #   Check() assertions     : add +2 to DBC_LEVEL
    #   Ensure() postconditions: add +4 to DBC_LEVEL
    #   Do not throw on error  : add +8 to DBC_LEVEL
    set( DRACO_DBC_LEVEL "7" )
    if( NOT CMAKE_CONFIGURATION_TYPES )
      if( "${Draco_BUILD_TYPE}" MATCHES "RELEASE" )
        set( DRACO_DBC_LEVEL "0" )
      elseif( "${Draco_BUILD_TYPE}" MATCHES "RELWITHDEBINFO" )
        set( DRACO_DBC_LEVEL "15" )
      endif()
    endif()
    set( DRACO_DBC_LEVEL "${DRACO_DBC_LEVEL}" CACHE STRING
      "Design-by-Contract (0-31)?" )
    # provide a constrained drop down menu in cmake-gui
    set_property( CACHE DRACO_DBC_LEVEL PROPERTY STRINGS
      0 1 2 3 4 5 6 7 9 10 11 12 13 14 15
      16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 )
  endif()

  if( CMAKE_CONFIGURATION_TYPES )
    # This generator expression will be expanded when the project is installed
    # (CMake-3.4.0+)
    set(DBSCFGDIR "\$<CONFIG>/" CACHE STRING
      "Install subdirectory for multiconfig build tools.")
    # Generate a complete installation directory structure to avoid errors of
    # the form "imported target includes non-existent path" when configuring
    # Jayenne.
    foreach( config ${CMAKE_CONFIGURATION_TYPES} )
      file( MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${config}/include )
    endforeach()
  endif()

  # ----------------------------------------
  # STATIC or SHARED libraries?
  # ----------------------------------------

  # Library type to build
  # Linux: STATIC is a lib<XXX>.a
  #        SHARED is a lib<XXX>.so (requires rpath or .so found in
  #               $LD_LIBRARY_PATH)
  # MSVC : STATIC is <XXX>.lib
  #        SHARED is <XXX>.dll (requires dll to be in $PATH or in same
  #               directory as exe).
  if( NOT DEFINED DRACO_LIBRARY_TYPE )
    set( DRACO_LIBRARY_TYPE "SHARED" )
  endif()
  set( DRACO_LIBRARY_TYPE "${DRACO_LIBRARY_TYPE}" CACHE STRING
    "Keyword for creating new libraries (STATIC or SHARED).")
  # Provide a constrained drop down list in cmake-gui.
  set_property( CACHE DRACO_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)

  # Enable parallel build for Eclipse:
  cmake_host_system_information( RESULT logical_cores QUERY NUMBER_OF_LOGICAL_CORES )
  set( CMAKE_ECLIPSE_MAKE_ARGUMENTS "-j ${logical_cores}" )

  # Set RPATH for all libraries on Apple platform
  if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
  endif()

  if( "${DRACO_LIBRARY_TYPE}" MATCHES "SHARED" )
     # Set replacement RPATH for installed libraries and executables
     # See http://www.cmake.org/Wiki/CMake_RPATH_handling

     # Do not skip the full RPATH for the build tree
     set( CMAKE_SKIP_BUILD_RPATH OFF )
     # When building, don't use the install RPATH already (but later on when
     # installing)
     set( CMAKE_BUILD_WITH_INSTALL_RPATH OFF )

     # For libraries created within the build tree, replace the RPATH
     # in the installed files with the install location.
     set( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" CACHE PATH
       "RPATH to embed in dynamic libraries and executables when
targets are installed." FORCE )

     # add the automatically determined parts of the RPATH which point to
     # directories outside the build tree to the install RPATH
     set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )
  endif()

endmacro()

##---------------------------------------------------------------------------##
## dbsInitExportTargets
##
## These fields are constructed during Draco configure and are
## saved/installed to lib/cmake/draco-X.X/draco-config.cmake.
##---------------------------------------------------------------------------##
macro( dbsInitExportTargets PREFIX )
  # Data for exporting during install
   set( ${PREFIX}_LIBRARIES ""  CACHE INTERNAL "List of draco targets" FORCE)
   set( ${PREFIX}_PACKAGE_LIST ""  CACHE INTERNAL
      "List of known package targets" FORCE)
   set( ${PREFIX}_TPL_LIST ""  CACHE INTERNAL
      "List of third party libraries known by this package" FORCE)
   set( ${PREFIX}_TPL_INCLUDE_DIRS ""  CACHE
      INTERNAL "List of include paths used by this package to find thrid party vendor header files."
      FORCE)
   set( ${PREFIX}_TPL_LIBRARIES ""  CACHE INTERNAL
      "List of third party libraries used by this package." FORCE)
endmacro()

#------------------------------------------------------------------------------#
# Save some build parameters for later use by --config options
#------------------------------------------------------------------------------#
macro( dbsConfigInfo )

   set( DBS_OPERATING_SYSTEM "${CMAKE_SYSTEM_NAME}")
   set( DBS_OPERATING_SYSTEM_VER "${CMAKE_SYSTEM}")

   # Suppliment with system commands as needed:
   if( UNIX )

      # Get some extra version information if this is RedHat.
      if( EXISTS "/etc/redhat-release" )
         file( READ "/etc/redhat-release" redhat_version )
         string( STRIP "${redhat_version}" redhat_version )
         set( DBS_OPERATING_SYSTEM_VER "${redhat_version} (${CMAKE_SYSTEM})" )
      endif( EXISTS "/etc/redhat-release" )

      # How many local cores
      if( EXISTS "/proc/cpuinfo" )
         file( READ "/proc/cpuinfo" cpuinfo )
         # string( STRIP "${cpuinfo}" cpuinfo )
         # convert one big string into a set of strings, one per line
         string( REGEX REPLACE "\n" ";" cpuinfo ${cpuinfo} )
         set( proc_ids "" )
         foreach( line ${cpuinfo} )
            if( ${line} MATCHES "processor" )
               list( APPEND proc_ids ${line} )
            endif()
         endforeach()
         list( LENGTH proc_ids DRACO_NUM_CORES )
         set( MPIEXEC_MAX_NUMPROCS ${DRACO_NUM_CORES} CACHE STRING
            "Number of cores on the local machine." )
      endif()

   elseif() # WIN32

      # OS version information
      # Windows XP
      GET_FILENAME_COMPONENT( win_prod_name "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion;ProductName]" NAME )
      GET_FILENAME_COMPONENT( win_sp "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion;CSDVersion]" NAME )
      get_filename_component( win_ver "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion;CurrentVersion]" NAME )
      get_filename_component( win_build "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion;CurrentBuildNumber]" NAME )
      get_filename_component( win_buildlab "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion;BuildLab]" NAME )

      set( DBS_OPERATING_SYSTEM "${win_prod_name}" )
      set( DBS_OPERATING_SYSTEM_VER "${win_ver}.${win_build} (${win_buildlab})" )
      if( EXISTS "C:/Program Files (x86)" )
         set( windows_bits "x64" )
         set( DBS_OPERATING_SYSTEM "${win_prod_name} (${windows_bits}) ${win_sp}" )
      else()
         set( DBS_OPERATING_SYSTEM "${win_prod_name} ${win_sp}" )
      endif()

      # Did we build 32-bit or 64-bit code?
      execute_process(
         COMMAND ${CMAKE_CXX_COMPILER}
         ERROR_VARIABLE tmp
         OUTPUT_STRIP_TRAILING_WHITESPACE
         OUTPUT_QUIET
         )
      string( REGEX REPLACE ".*for ([0-9x]+)" "\\1"
         tmp "${CMAKE_CXX_COMPILER_VER}" )
      if( ${tmp} MATCHES "80x86" )
         set( DBS_ISA_MODE "32-bit" )
      elseif( ${tmp} MATCHES "x64" )
         set( DBS_ISA_MODE "64-bit" )
      endif( ${tmp} MATCHES "80x86" )

      # Try to get the build machine name
      execute_process(
         COMMAND "c:/windows/system32/ipconfig.exe" "/all"
         OUTPUT_VARIABLE windows_ip_configuration
         )
      string( REGEX REPLACE ".*Host Name[.: ]+([A-z]+).*Primary.*" "\\1" DBS_TARGET "${windows_ip_configuration}" )

   endif()

endmacro()

#------------------------------------------------------------------------------#
# End
#------------------------------------------------------------------------------#
