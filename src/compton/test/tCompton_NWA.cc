//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   cdi_eospac/test/tEospac.cc
 * \author Kelly Thompson
 * \date   Mon Apr 2 14:20:14 2001
 * \brief  Implementation file for tEospac
 * \note   Copyright (C) 2016 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#include "compton/Compton_NWA.hh"
#include "ds++/Release.hh"
#include "ds++/SP.hh"
#include "ds++/ScalarUnitTest.hh"
#include "ds++/Soft_Equivalence.hh"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace rtt_compton_test {

using rtt_dsxx::SP;
using rtt_dsxx::soft_equiv;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

//!  Tests the Eospac constructor and access routines.
void compton_file_test(rtt_dsxx::UnitTest &ut) {
  // Start the test.

  std::cout << "\nTest of C++ code calling NWA routines\n" << std::endl;

  ITFAILS;
}
}

//---------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
  rtt_dsxx::ScalarUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    // >>> UNIT TESTS
    rtt_compton_test::compton_file_test(ut);
  }
  UT_EPILOG(ut);
}
