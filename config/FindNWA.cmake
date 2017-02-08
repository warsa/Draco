# - Find NWA
# Find the native NWA includes and library
#
#  NWA_INCLUDE_DIRS   - where to find eos_Interface.h, etc.
#  NWA_LIBRARIES      - List of libraries when using NWA.
#  NWA_FOUND          - True if NWA found.

if (APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.dylib")
endif()

 find_path( NWA_INCLUDE_DIR 
    NAMES
       multigroup_lib_builder.hh
    PATHS
       ${NWA_INC_DIR}
       $ENV{NWA_INC_DIR}
       $ENV{VENDOR_DIR}/include
       ${VENDOR_DIR}/include
       /home/keadyk/development/CSK_generator/src
    NO_DEFAULT_PATH
)

if( WIN32 )
   if( NWA_STATIC )
      set( NWA_LIBRARY_NAME libcompton.lib)
   else()
      set( NWA_LIBRARY_NAME libcompton_dll.lib)
   endif()
else()
   set( NWA_LIBRARY_NAME libLib_compton.a)
endif()

find_library(NWA_LIBRARY
    NAMES ${NWA_LIBRARY_NAME}
    PATHS
        ${NWA_LIB_DIR}
        $ENV{NWA_LIB_DIR}
        $ENV{VENDOR_DIR}/lib
        ${VENDOR_DIR}/lib
        /home/keadyk/builds/gcc/CSK_generator
    NO_DEFAULT_PATH
)

# If above fails, look in default locations
if( NOT NWA_LIBRARY )
   find_path( NWA_INCLUDE_DIR    NAMES multigroup_lib_builder.hh )
   find_library(NWA_LIBRARY      NAMES ${NWA_LIBRARY_NAME} )
endif()
mark_as_advanced( NWA_LIBRARY NWA_INCLUDE_DIR NWA_LIBRARY )

# handle the QUIETLY and REQUIRED arguments and set NWA_FOUND to TRUE if 
# all listed variables are TRUE (removed NWA_INCLUDE_DIR)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NWA DEFAULT_MSG NWA_LIBRARY)

if (NWA_FOUND)
   set(NWA_FOUND ${NWA_FOUND} CACHE BOOL "Did we find the NWA libraries?")
   set(NWA_INCLUDE_DIRS ${NWA_INCLUDE_DIR})
   set(NWA_LIBRARIES    ${NWA_LIBRARY} CACHE
      FILEPATH "NWA libraries for linking."  )
   
   string( REPLACE "_dll.lib" ".dll" NWA_DLL ${NWA_LIBRARY} )
   mark_as_advanced( NWA_DLL )
   if( EXISTS ${NWA_DLL} )
      set(NWA_DLL_LIBRARIES "${NWA_DLL}" CACHE STRING 
         "list of compton dll files.")
   mark_as_advanced( NWA_DLL_LIBRARIES NWA_LIBRARIES )
   else()
      set( NWA_DLL "NOTFOUND")
   endif()
endif()

if( VERBOSE )
message("
NWA_FOUND =        ${NWA_FOUND}
NWA_INCLUDE_DIRS = ${NWA_INCLUDE_DIR}
NWA_LIBRARIES    = ${NWA_LIBRARY}
")
endif()
