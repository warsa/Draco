//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/XGetopt.cc
 * \author Katherine Wang
 * \date   Wed Nov 10 09:35:09 2010
 * \brief  Command line argument handling similar to getopt.
 * \note   Copyright (C) 2015 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iterator>
#include <vector>
#include <map>
#include <list>
#include "XGetopt.hh"

namespace rtt_dsxx
{

//---------------------------------------------------------------------------//
//  getopt
//
//  Return code >0 if argv contains known shortopts.
//---------------------------------------------------------------------------//

using std::string;

char    *optarg;                // global argument pointer
int      optind = 1;            // global argv index (set to 1, to skip exe name)

int getopt( int argc, char ** argv, std::string const & shortopts,
            longopt_map map )
{
    // convert argv into a vector<string>...
    std::vector< std::string > options( argv, argv + argc );

    // convert shortopts into vector<char> + vector<int>...
    std::vector< char> vshortopts;
    std::vector< int > vshortopts_hasarg; // map<char,bool>
    //std::vector< int > rtrns;
    for( size_t i = 0; i < shortopts.size(); ++i )
    {
        vshortopts.push_back( shortopts[ i ] );
        vshortopts_hasarg.push_back( 0 ); // assume no required argument.
        if( i + 1 < shortopts.size() && shortopts[ i + 1 ] == std::string( ":" )[ 0 ] )
        {
            vshortopts_hasarg[ i ] = 1;
            ++i;
        }
    }

    // Look for command line arguments that match provided list:
    for( ; optind < /*static_cast<int>*/( options.size() ); ++optind )
    {
        // stop processing command line arguments

        if( options[ optind ] == std::string( "--" ) )

        {
            return -1;
        }

        // consider string-based optons here.
        if( options[optind].substr(0,2) == std::string("--") )
	{
	    for( longopt_map::iterator it=map.begin(); it!=map.end(); ++it )
       	    {
               //std::cout << it->first << " :: " << it->second << std::endl; //?
	   
	       if( options[optind] == std::string("--")+(it->first))
	       {
                  // what is the short arg equivalent.
	          size_t j(0);
                  for( ; j<vshortopts.size(); ++j )
		     if( std::string(1,vshortopts[j]) == std::string(1,it->second) ) 
                        break;

                    if( vshortopts_hasarg[ j ] == 1 )
                    {
                        ++optind;
                        optarg = argv[ optind ];
                    }
                    ++optind;
                    std::cout << std::endl;
                    return vshortopts[ j ];
                }

            }

        }

	//size_t j=0;
        // consider single letter options here.
	//std::string tst = options[optind];
        for( size_t j=0; j<vshortopts.size(); ++j )
        {

            if( options[optind] == std::string("-")+string(1,vshortopts[j]) )
            {  
                if( vshortopts_hasarg[j] == 1 )
                {
                   ++optind;
                   optarg = argv[optind];
                 }
                 ++optind;
                 std::cout << std::endl;
                 return vshortopts[j];
            }
	}

	if( options[optind].substr(0,1) == std::string("-") )
	{  
	    while( options[optind].size() > 2 )
	    {	
		size_t j=0;
	        for( ; j<vshortopts.size(); ++j )
		    if( options[optind].substr(1, 1) == string(1,vshortopts[j]) )
		        break;

		std::cout<<std::endl;
		std::cout<<optind<<std::endl;
		std::cout<<options.size()<<std::endl;
                std::string newarg = ( std::string("-") + options[optind].substr(2,options[optind].size()) );
		// options.push_back( "-" + options[optind].substr(2,options[optind].size()) );
		std::cout<<argv[optind]<<std::endl;		    

		// ++optind;
		// argv[ optind ] = (char*&)options[ optind ];
		argv[ optind ] = (char*&)newarg;
		// std::cout<<options[optind]<<std::endl;
		std::cout<<newarg<<std::endl;
		std::cout<<argv[optind]<<std::endl;
		std::cout<<argv[0]<<std::endl;
		std::cout<<argv[1]<<std::endl;
		std::cout<<argv[2]<<std::endl;

		std::cout<<optind<<std::endl;
		std::cout<<options.size()<<std::endl;

	    	return vshortopts[j];
	    }
	}
    }   

    return -1;
}

} // end namespace rtt_dsxx

//---------------------------------------------------------------------------//
// end of XGetopt.cc
//---------------------------------------------------------------------------//
