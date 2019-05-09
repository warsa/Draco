#-----------------------------*-cmake-*----------------------------------------#
# file   config/unix-g++.cmake
# brief  Establish flags for Unix/Linux - Gnu C++
# note   Copyright (C) 2016-2019 Triad National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#

include_guard(GLOBAL)

# Note: In config/compilerEnv.cmake, the build system sets flags for
# 1) the language standard (C++14, C99, etc)
# 2) interprocedural optimization.

# Notes:
# ----------------------------------------
# Useful options that could be added to aid debugging
# http://gcc.gnu.org/wiki/Atomic/GCCMM
# http://gcc.gnu.org/wiki/TransactionalMemory
# http://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://gcc.gnu.org/gcc-5/changes.html
# http://stackoverflow.com/questions/3375697/useful-gcc-flags-for-c

#
# Declare CMake options related to GCC
#
if( NOT DEFINED GCC_ENABLE_ALL_WARNINGS )
  option( GCC_ENABLE_ALL_WARNINGS "Add \"-Weffc++\" to the compile options."
    OFF )
endif()
if( NOT DEFINED GCC_ENABLE_GLIBCXX_DEBUG )
option( GCC_ENABLE_GLIBCXX_DEBUG
  "Use special version of libc.so that includes STL bounds checking." OFF )
endif()

#
# Compiler flag checks
#
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
check_c_compiler_flag(   "-march=native" HAS_MARCH_NATIVE )
if( DEFINED CMAKE_CXX_COMPILER_ID )
  check_cxx_compiler_flag( "-Wnoexcept"    HAS_WNOEXCEPT )
  check_cxx_compiler_flag( "-Wsuggest-attribute=const" HAS_WSUGGEST_ATTRIBUTE )
  check_cxx_compiler_flag( "-Wunused-local-typedefs"
    HAS_WUNUSED_LOCAL_TYPEDEFS )
  #check_cxx_compiler_flag( "-Wunused-macros" HAS_WUNUSED_MACROS )
  #check_cxx_compiler_flag( "-Wzero-as-null-pointer-constant"
  #  HAS_WZERO_AS_NULL_POINTER_CONSTANT )
endif()

# is this bullseye?
execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} --version
  ERROR_VARIABLE DBS_CXX_IS_BULLSEYE
  OUTPUT_VARIABLE tmp
  )

if( ${DBS_CXX_IS_BULLSEYE} MATCHES BullseyeCoverage )
  set( DBS_CXX_IS_BULLSEYE "ON" )
  set( DBS_CXX_IS_BULLSEYE ${DBS_CXX_IS_BULLSEYE}
    CACHE BOOL "Are we running Bullseye" )
  mark_as_advanced( DBS_CXX_IS_BULLSEYE )
endif()

# LTO (option '-flto') is controlled via the cmake variable
# CMAKE_INTERPROCEDURAL_OPTIMIZATION, see config/compilerEnv.cmake.

#
# Compiler Flags
#
# Consider using these diagnostic flags for Debug builds:
# -Wcast-qual - warn about casts that remove qualifiers like const.
# -Wstrict-overflow=4
# -Wwrite-strings
# -Wunreachable-code
# - https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
#
# Consider using these optimization flags:
# -ffast-math -ftree-vectorize
# -fno-finite-math-only -fno-associative-math -fsignaling-nans
#
# Added, but shouldn't be needed:
# -Wno-expansion-to-defined - unable to use GCC diagnostic pragma to suppress
#           warnings.

if( NOT CXX_FLAGS_INITIALIZED )
  set( CXX_FLAGS_INITIALIZED "yes" CACHE INTERNAL "using draco settings." )

  set( CMAKE_C_FLAGS                "-Wcast-align -Wpointer-arith -Wall -pedantic" )
  if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 )
    string( APPEND CMAKE_C_FLAGS    " -Wno-expansion-to-defined -Wnarrowing" )
  endif()
  set( CMAKE_C_FLAGS_DEBUG          "-g -gdwarf-3 -fno-inline -fno-eliminate-unused-debug-types -O0 -Wextra -Wundef -Wunreachable-code -DDEBUG")
  # -Wfloat-equal
  # -Werror
  # -Wconversion
  # -Winline
  set( CMAKE_C_FLAGS_RELEASE        "-O3 -funroll-loops -D_FORTIFY_SOURCE=2 -DNDEBUG" )
  set( CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_RELEASE}" )
  set( CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -gdwarf-3 -fno-eliminate-unused-debug-types -Wextra -Wno-expansion-to-defined -funroll-loops" )

  if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 )
    # See https://gcc.gnu.org/gcc-5/changes.html
    # UndefinedBehaviorSanitizer gained a few new sanitization options:
    #  -fsanitize=alignment: enable alignment checking, detect various
    #             misaligned objects;
    #  -fsanitize=object-size: enable object size checking, detect various
    #             out-of-bounds accesses.
    #  -fsanitize=vptr: enable checking of C++ member function calls, member
    #             accesses and some conversions between pointers to base and
    #             derived classes, detect if the referenced object does not have
    #             the correct dynamic type.
    if( NOT DEFINED ENV{TRAVIS} )
      # Some options (including these) seem to confuse Travis.
      # See https://stackoverflow.com/questions/50024731/ld-unrecognized-option-push-state-no-as-needed
      string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=float-divide-by-zero")
      string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=float-cast-overflow")
    endif()
    string( APPEND CMAKE_C_FLAGS_DEBUG " -fdiagnostics-color=auto")
#    string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=vptr")
#    string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=object-size")
#    string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=alignment")
#    string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=address")
    # - The '-fsanitize=leak' option sounds great but it finds too many issues in
    #   OpenMPI.
    # string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=leak")
    # GCC_COLORS="error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01"
  endif()
  if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0 )
    # See https://gcc.gnu.org/gcc-6/changes.html
    # -fsanitize=bounds-strict, which enables strict checking of array
    #            bounds. In particular, it enables -fsanitize=bounds as well as
    #            instrumentation of flexible array member-like arrays.
  endif()
  if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 )
    # See https://gcc.gnu.org/gcc-7/changes.html
    # -fsanitize-address-use-after-scope: sanitation of variables whose address
    #            is taken and used after a scope where the variable is
    #            defined. On by default when -fsanitize=address.
    # -fsanitize=signed-integer-overflow
    # -Wduplicated-branches warns when an if-else has identical branches.
    if( NOT DEFINED ENV{TRAVIS} )
      string( APPEND CMAKE_C_FLAGS_DEBUG " -fsanitize=signed-integer-overflow")
    endif()
  endif()
  if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0 )
    # See https://gcc.gnu.org/gcc-8/changes.html
    # -fsanitize=pointer-compare warn about subtraction (or comparison) of
    #                  pointers that point to a different memory object.
    # -fsanitize=pointer-subtract
    string( APPEND CMAKE_CXX_FLAGS_DEBUG " -Wold-style-cast")
    string( APPEND CMAKE_CXX_FLAGS_DEBUG " -fdiagnostics-show-template-tree")
  endif()

  # [2017-04-15 KT] -march=native doesn't seem to work correctly on toolbox
  # Systems running CRAY_PE use commpile wrappers to specify this option.
  site_name( sitename )
  string( REGEX REPLACE "([A-z0-9]+).*" "\\1" sitename ${sitename} )
  if (HAS_MARCH_NATIVE AND
      NOT APPLE AND
      NOT CRAY_PE AND
      NOT "${sitename}" MATCHES "toolbox")
    string( APPEND CMAKE_C_FLAGS " -march=native" )
  endif()

  set( CMAKE_CXX_FLAGS                "${CMAKE_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG} -Woverloaded-virtual")
  set( CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE}")
  set( CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_RELEASE}")
  set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}" )

  # Extra Debug flags that only exist in newer gcc versions.
  if( HAS_WNOEXCEPT )
    string( APPEND CMAKE_CXX_FLAGS_DEBUG " -Wnoexcept" )
  endif()
  if( HAS_WSUGGEST_ATTRIBUTE )
    string( APPEND CMAKE_CXX_FLAGS_DEBUG " -Wsuggest-attribute=const" )
  endif()
  if( HAS_WUNUSED_LOCAL_TYPEDEFS )
    string( APPEND CMAKE_CXX_FLAGS_DEBUG " -Wunused-local-typedefs" )
  endif()

endif()

##---------------------------------------------------------------------------##
# Ensure cache values always match current selection
##---------------------------------------------------------------------------##
set( CMAKE_C_FLAGS                "${CMAKE_C_FLAGS}"                CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_C_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG}"          CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE}"        CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_MINSIZEREL}"     CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}" CACHE
     STRING "compiler flags" FORCE )

set( CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS}"                CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}"          CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}"        CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL}"     CACHE
     STRING "compiler flags" FORCE )
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE
     STRING "compiler flags" FORCE )

#
# Toggle compiler flags for optional features
#
toggle_compiler_flag( GCC_ENABLE_ALL_WARNINGS "-Weffc++" "CXX" "DEBUG")
toggle_compiler_flag( GCC_ENABLE_GLIBCXX_DEBUG
  "-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" "CXX" "DEBUG" )
toggle_compiler_flag( OPENMP_FOUND ${OpenMP_C_FLAGS} "C;CXX" "" )

# Issues with tstFMA[12].cc:
# toggle_compiler_flag( HAS_WUNUSED_MACROS "-Wunused-macros" "C;CXX" "" )

# On SQ, our Random123/1.08 vendor uses a series of include directives that fail
# to compile with g++-4.7.2 when the -pedantic option is requested. The core
# issue is that fabs is defined with two different exception signatures in
# math.h and in ppu_intrinsics.h. On this platform, we choose not use -pedantic.
if( ${SITENAME} MATCHES "seq" )
  toggle_compiler_flag( OFF "-pedantic" "CXX" "")
endif()

#------------------------------------------------------------------------------#
# End config/unix-g++.cmake
#------------------------------------------------------------------------------#
