//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   c4/parallel_unit_test.i.hh
 * \author Kent Grimmett Budge
 * \date   Tue Nov  6 13:12:37 2018
 * \brief  Member definitions of class test
 * \note   Copyright (C) TRIAD, LLC. All rights reserved. */
//----------------------------------------------------------------------------//

#ifndef dsxx_parallel_unit_test_i_hh
#define dsxx_parallel_unit_test_i_hh

#include "c4/ParallelUnitTest.hh"

namespace rtt_dsxx {

template <typename Lambda, typename Release>
int do_parallel_unit_test(int argc, char *argv[], Release release,
                          Lambda const &lambda) {
  rtt_c4::ParallelUnitTest ut(argc, argv, release);
  try {
    lambda(ut);
  }
  UT_EPILOG(ut);
}

} // end namespace rtt_dsxx

#endif // dsxx_test_i_hh

//----------------------------------------------------------------------------//
// end of ds++/parallel_unit_test.i.hh
//----------------------------------------------------------------------------//
