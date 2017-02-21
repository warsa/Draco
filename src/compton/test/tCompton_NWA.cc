//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   compton/test/tCompton_NWA.cc
 * \author Kendra Keady
 * \date   Feb 10
 * \brief  Implementation file for tCompton_NWA
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

//!  Tests the Compton_NWA constructor and a couple of access routines.
void compton_file_test(rtt_dsxx::UnitTest &ut) {
  // Start the test.

  std::cout << "\n------------------------------------------" << std::endl;
  std::cout << "   Test Draco code calling NWA routines" << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  // open a small mg opacity file:
  std::string filename = "mg_ascii.compton";
  std::cout << "Attempting to construct a Compton_NWA object...\n" << std::endl;
  std::shared_ptr<rtt_compton::Compton_NWA> compton_test;

  try {
    compton_test.reset(new rtt_compton::Compton_NWA(filename));
  } catch (int asrt) {
    FAILMSG("Failed to construct a Compton_NWA object!");
    // if construction fails, there is no reason to continue testing...
    return;
  }
  std::cout << "\n(...Success!)" << std::endl;

  // Check some of the data in the NWA-opened file:
  std::vector<double> grp_bds = compton_test->get_group_bounds();

  Ensure(grp_bds.size() == 2);

  if (!soft_equiv(grp_bds[0], 3.91389000e-02)) {
    std::cout << "Lower group bound read incorrectly!" << std::endl;
    ITFAILS;
  }
  if (!soft_equiv(grp_bds[1], 5.87084000e-02)) {
    std::cout << "Upper group bound read incorrectly!" << std::endl;
    ITFAILS;
  }

  if (!soft_equiv(compton_test->get_min_etemp(), 1e-6)) {
    std::cout << "Min etemp read incorrectly!" << std::endl;
    ITFAILS;
  }
  if (!soft_equiv(compton_test->get_max_etemp(), 7e-4)) {
    std::cout << "Max etemp read incorrectly!" << std::endl;
    ITFAILS;
  }

  if (ut.numFails == 0) {
    std::cout << "\nCorrectly read group bounds and electron temps!"
              << std::endl;
  }

  // try "interpolating" at one of the exact eval points in the test library,
  // and check the result:
  std::vector<std::vector<std::vector<double>>> interp_data =
      compton_test->interpolate(4.87227167e-04);

  // Check the size of the returned data:
  Ensure(interp_data.size() == 1);
  Ensure(interp_data[0].size() == 1);
  Ensure(interp_data[0][0].size() == 4);

  // Check that the data is actually correct:
  if (!soft_equiv(interp_data[0][0][0], 4.45668383e+00))
    ITFAILS;
  if (!soft_equiv(interp_data[0][0][1], 3.17337784e-01))
    ITFAILS;
  if (!soft_equiv(interp_data[0][0][2], 4.50133379e-01))
    ITFAILS;
  if (!soft_equiv(interp_data[0][0][3], 3.59663442e-02))
    ITFAILS;

  if (ut.numFails == 0) {
    std::cout << "\nCorrectly read multigroup data points!" << std::endl;
  }

  if (ut.numFails == 0) {
    PASSMSG("Successfully linked Draco against NWA.");
  } else {
    FAILMSG("Did not successfully link Draco against NWA.");
  }
}

//!  Tests the Compton_NWA mg build capability
void compton_build_test(rtt_dsxx::UnitTest &ut) {
  // Start the test.

  std::cout << "\n------------------------------------------" << std::endl;
  std::cout << "Test Draco call to NWA mg opacity builder" << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  // open a small pointwise opacity file:
  std::string filename = "lagrange_csk_ascii.compton";
  std::cout << "Attempting to construct a Compton_NWA object..." << std::endl;

  // make an uninitialized pointer...
  std::shared_ptr<rtt_compton::Compton_NWA> compton_test;

  // make a small fake group structure to pass in:
  std::vector<double> test_groups(5, 0.0);
  test_groups[0] = 20.0;
  test_groups[1] = 30.0;
  test_groups[2] = 40.0;
  test_groups[3] = 50.0;
  test_groups[4] = 60.0;

  // set the number of angular points to retrieve (legendre or otherwise)
  size_t nxi = 3;

  try {
    // (This call has some output of its own, so we print some newlines
    // around it)
    std::cout << "\n\n";
    compton_test.reset(
        new rtt_compton::Compton_NWA(filename, test_groups, nxi));
    std::cout << "\n\n";
  } catch (rtt_dsxx::assertion &asrt) {
    FAILMSG("Failed to construct a Compton_NWA object!");
    // if construction fails, there is no reason to continue testing...
    return;
  }
  std::cout << "(...Success!)" << std::endl;

  // Check some of the data in the NWA-formed data file:
  std::vector<double> grp_bds = compton_test->get_group_bounds();

  // Check that the stored group structure is correct:
  if (!soft_equiv(grp_bds[0], test_groups[0]))
    ITFAILS;
  if (!soft_equiv(grp_bds[1], test_groups[1]))
    ITFAILS;
  if (!soft_equiv(grp_bds[2], test_groups[2]))
    ITFAILS;
  if (!soft_equiv(grp_bds[3], test_groups[3]))
    ITFAILS;
  if (!soft_equiv(grp_bds[4], test_groups[4]))
    ITFAILS;

  if (!soft_equiv(compton_test->get_min_etemp(), 0.0)) {
    std::cout << "Min etemp read incorrectly!" << std::endl;
    ITFAILS;
  }
  if (!soft_equiv(compton_test->get_max_etemp(), 1.0)) {
    std::cout << "Max etemp read incorrectly!" << std::endl;
    ITFAILS;
  }

  if (ut.numFails == 0) {
    std::cout << "\nCorrectly stored group bounds and electron temps! "
              << std::endl;
  }

  if (ut.numFails == 0) {
    PASSMSG("Successfully built an NWA mg library.");
  } else {
    FAILMSG("Did not successfully build an NWA mg library.");
  }
}

//!  Tests Compton_NWA's error-handling on a non-existent file.
void compton_fail_test(rtt_dsxx::UnitTest &ut) {
  std::cout << "\n------------------------------------------" << std::endl;
  std::cout << "   Test NWA error-handling for bad file" << std::endl;
  std::cout << "------------------------------------------" << std::endl;
  // open a small mg opacity file:
  std::string filename = "non_existent.compton";
  std::cout << "Testing with a non-existent file...\n" << std::endl;
  std::shared_ptr<rtt_compton::Compton_NWA> compton_test;

  bool caught = false;
  try {
    compton_test.reset(new rtt_compton::Compton_NWA(filename));
  } catch (int asrt) {
    // We successfully caught the bad file!
    caught = true;
  }

  if (!caught)
    ITFAILS;

  if (ut.numFails == 0) {
    PASSMSG("Successfully caught an NWA exception.");
  } else {
    FAILMSG("Did not successfully catch an NWA exception.");
  }
}
}

//---------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
  rtt_dsxx::ScalarUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    // >>> UNIT TESTS
    rtt_compton_test::compton_file_test(ut);
    rtt_compton_test::compton_build_test(ut);
    rtt_compton_test::compton_fail_test(ut);
  }
  UT_EPILOG(ut);
}
