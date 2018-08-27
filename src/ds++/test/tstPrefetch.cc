//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/test/tstPrefetch.cc
 * \brief  Demonstrate/Test the prefetch function.
 * \note   Copyright (C) 2018 Los Alamos National Security, LLC.
 *         All rights reserved */
//---------------------------------------------------------------------------//

#include <cmath>
#include <iostream>

#include "ds++/Prefetch.hh"
#include "ds++/Release.hh"
#include "ds++/ScalarUnitTest.hh"

using namespace std;
using namespace rtt_dsxx;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
void runtest(rtt_dsxx::UnitTest &ut) {
  cout << "start:" << endl;

  // Loop without prefetch
  unsigned const R = 300;
  unsigned const N = 1000000;
  vector<double> huge(N);
  double sum = 0.0;
  for (unsigned k = 0; k < R; ++k) {
    for (unsigned i = 0; i < N; i += CACHE_LINE_DOUBLE) {
      for (unsigned j = 0; j < CACHE_LINE_DOUBLE; ++j) {
        huge[j] = 0.1 * j;
      }
    }
    for (unsigned i = 0; i < N; i += CACHE_LINE_DOUBLE) {
      for (unsigned j = 0; j < CACHE_LINE_DOUBLE; j += 2) {
        sum += sqrt(huge[j] * huge[j] + huge[j + 1] * huge[j + 1]);
      }
    }
  }
  cout << sum << endl;

  // Loop with prefetch
  for (unsigned k = 0; k < R; ++k) {
    for (unsigned i = 0; i < N; i += CACHE_LINE_DOUBLE) {
      prefetch_cache_line(&huge[i + CACHE_LINE_DOUBLE], 1, 0);
      for (unsigned j = 0; j < CACHE_LINE_DOUBLE; ++j) {
        huge[j] = 0.12 * j;
      }
    }
    for (unsigned i = 0; i < N; i += CACHE_LINE_DOUBLE) {
      prefetch_cache_line(&huge[i + CACHE_LINE_DOUBLE], 0, 1);
      for (unsigned j = 0; j < CACHE_LINE_DOUBLE; j += 2) {
        sum += sqrt(huge[j] * huge[j] + huge[j + 1] * huge[j + 1]);
      }
    }
  }
  cout << sum << endl;

  ut.passes("run to completion"); // Nothing crashed
}

//---------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
  rtt_dsxx::ScalarUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    runtest(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
//  end of tstSafe_Divide.cc
//---------------------------------------------------------------------------//
