//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/SystemCall.cc
 * \brief  Implementation for the Draco wrapper for system calls. This 
           routine attempts to hide differences between Unix/Windows system 
           calls.
 * \note   Copyright (C) 2012 Los Alamos National Security, LLC.
 *         All rights reserved.
 * \version $Id$
 */
//---------------------------------------------------------------------------//

#include "SystemCall.hh"
#include "Assert.hh"

#include <cstring>      // strncpy()
#include <cerrno>       // errno
#include <cstdlib>      // _fullpath
#ifdef UNIX
#include <sys/param.h>  // MAXPATHLEN
#include <unistd.h>     // gethostname
#include <sys/stat.h>   // stat
#endif
#ifdef WIN32
#include <winsock2.h>  // gethostname
#include <process.h>   // _getpid
#include <direct.h>    // _getcwd
#include <sys/types.h> // _stat
#include <sys/stat.h>  // _stat
#endif

#include <iostream>

namespace rtt_dsxx
{

//===========================================================================//
// FREE FUNCTIONS
//===========================================================================//

//---------------------------------------------------------------------------//
/*! \brief Wrapper for system dependent hostname call.
 *
 * Windows:
 *     HOST_NAME_MAX set to MAX_COMPUTERNAME_LENGTH in config.h
 *
 * Catamount systems:
 *     HOST_NAME_MAX hard coded by CMake in config.h
 * 
 * Unix/Linux:
 *     HOST_NAME_MAX loaded from <climit>
 *
 * Mac OSX:
 *     HOST_NAME_MAX set to _POSIX_HOST_NAME_MAX in config.h
 */
std::string draco_gethostname( void )
{
    // Windows: gethostname from <winsock2.h>
#ifdef WIN32
    char hostname[HOST_NAME_MAX];
    int err = gethostname(hostname, sizeof(hostname));
    if (err) strncpy(hostname, "gethostname() failed", HOST_NAME_MAX);
    return std::string(hostname);
#endif    
    
    // Linux: gethostname from <unistd.h>
#ifdef HAVE_GETHOSTNAME 
    char hostname[HOST_NAME_MAX];
    int err = gethostname(hostname, HOST_NAME_MAX);
    if (err) strncpy(hostname, "gethostname() failed", HOST_NAME_MAX);
    return std::string(hostname);
    
    // Catamount systems do not have gethostname().
#else
    return std::string("Host (unknown)");
#endif

} // draco_hostname


//---------------------------------------------------------------------------//
/*! \brief Wrapper for system dependent pid call..
 *
 * Catamount systems do not have getpid().  This function will return -1.
 */
int draco_getpid(void)
{
#ifdef WIN32
    int i = _getpid();
    return i;
#else    
    
#ifdef HAVE_GETHOSTNAME 
    return getpid();
#else
    // Catamount systems do not have getpid().  This function will return -1.
    return -1;
#endif
#endif
} // draco_pid

//---------------------------------------------------------------------------//
/*! \brief Wrapper for system dependent getcwd call.
 *
 */
std::string draco_getcwd(void)
{
    // Identify the current working directory.
#ifdef WIN32
	char * buffer;
	Insist( (buffer = _getcwd(NULL, 0)) != NULL,
           std::string("getcwd failed: " + std::string(strerror(errno))));
	std::string curr_path( buffer, buffer+strnlen(buffer,MAXPATHLEN));
	free(buffer);
    return curr_path;
#else
    char curr_path[MAXPATHLEN]; curr_path[0] = '\0';
    Insist(getcwd(curr_path, MAXPATHLEN) != NULL,
           std::string("getcwd failed: " + std::string(strerror(errno))));
    return std::string( curr_path );
#endif
}

//---------------------------------------------------------------------------//
/*! \brief Wrapper for system dependent stat call.
 *
 * http://msdn.microsoft.com/en-us/library/14h5k7ff%28v=vs.80%29.aspx
 */
int draco_getstat( std::string const &  fqName )
{
#ifdef WIN32
    /*! \note If path contains the location of a directory, it cannot 
     * contain a trailing backslash. If it does, -1 will be returned and 
     * errno will be set to ENOENT. */
    std::string clean_fqName;
    if( fqName[fqName.size()-1] == rtt_dsxx::WinDirSep ||
        fqName[fqName.size()-1] == rtt_dsxx::UnixDirSep )
        clean_fqName=fqName.substr(0,fqName.size()-1);
    else
        clean_fqName=fqName;
    std::cout << "fqName = " << fqName 
            << "\nclean_fqName = " << clean_fqName << std::endl;
    struct _stat buf;
    return _stat(clean_fqName.c_str(), &buf);
#else
    struct stat buf;
    return stat(fqName.c_str(), &buf);
#endif
}

//---------------------------------------------------------------------------//
/*! \brief Wrapper for system dependent realpath call.
 *
 */
std::string draco_getrealpath( std::string const & path )
{
    char buffer[MAXPATHLEN]; // _MAX_PATH
#ifdef WIN32
    // http://msdn.microsoft.com/en-us/library/506720ff%28v=vs.100%29.aspx
    Insist(_fullpath( buffer, path.c_str(), MAXPATHLEN ) != NULL, "Invalid path." );
#else
	char *p;
    Insist((p = realpath(path.c_str(), buffer)) != NULL, "Invalid path." );
#endif
    return std::string(buffer);
}


} // end of rtt_dsxx

//---------------------------------------------------------------------------//
// end of SystemCall.cc
//---------------------------------------------------------------------------//