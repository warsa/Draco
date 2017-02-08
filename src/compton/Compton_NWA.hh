//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   compton/Compton_NWA.hh
 * \author Kendra Keady
 * \date   Mon Apr  2 14:14:29 2001
 * \brief  Header file for compton NWA interface -- linked against library
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#ifndef __compton_Compton_NWA_hh__
#define __compton_Compton_NWA_hh__

// C++ standard library dependencies
#include <iostream>
#include <memory>

namespace rtt_compton {

class Compton_NWA {
public:
  Compton_NWA(const std::string &);
};
}

#endif
