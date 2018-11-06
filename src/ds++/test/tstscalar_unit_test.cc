//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   ds++/test/tstscalar_unit_test.cc
 * \author Kent Grimmett Budge
 * \date   Tue Nov  6 13:19:40 2018
 * \brief  Test the test function for scalar case.
 * \note   Copyright (C) 2018 TRIAD, LLC. All rights reserved. */
//----------------------------------------------------------------------------//

#include "ds++/Release.hh"
#include "ds++/scalar_unit_test.hh"

using namespace rtt_dsxx;

//----------------------------------------------------------------------------//
// TESTS
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
  return do_scalar_unit_test(argc, argv, release,
                             [](UnitTest &ut) { ut.passes("basic run"); });
}

//----------------------------------------------------------------------------//
// end of tstscalar_unit_test.cc
//----------------------------------------------------------------------------//
