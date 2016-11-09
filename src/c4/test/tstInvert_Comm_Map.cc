//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/test/tstInvert_Comm_Map.cc
 * \author Mike Buksas, Rob Lowrie
 * \date   Mon Nov 19 16:33:08 2007
 * \brief  Tests Invert_Comm_Map
 * \note   Copyright (C) 2016 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//

#include "c4/Invert_Comm_Map.hh"
#include "c4/ParallelUnitTest.hh"
#include "ds++/Release.hh"

using namespace std;
using namespace rtt_c4;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
void test2(rtt_c4::ParallelUnitTest &ut) {
  size_t const node = rtt_c4::node();
  std::vector<int> to_nodes;

  if (node == 0) {
    to_nodes.resize(1);
    to_nodes[0] = 1;
  }

  if (node == 1) {
    // No communication from node 1.
    to_nodes.resize(0);
  }

  std::vector<int> from_nodes(0);
  invert_comm_map(to_nodes, from_nodes);

  if (node == 0) {
    if (from_nodes.size() != 0u)
      FAILMSG("Incorrect map size on node 0.");
  }

  if (node == 1) {
    if (from_nodes.size() != 1u)
      FAILMSG("Incorrect size of map on node 1.");
    if (from_nodes[0] != 0)
      FAILMSG("Incorrect map contents on node 1.");
  }

  if (ut.numFails == 0)
    ut.passes("test2 passes");
  else
    ut.failure("test2 failed");

  return;
}

//----------------------------------------------------------------------------//
void test4(rtt_c4::ParallelUnitTest &ut) {
  size_t const node = rtt_c4::node();
  std::vector<int> to_nodes;

  if (node == 0) {
    to_nodes.push_back(1);
    to_nodes.push_back(2);
    to_nodes.push_back(3);
  }
  if (node == 1) {
    to_nodes.push_back(0);
  }
  if (node == 2) {
    to_nodes.push_back(0);
  }
  if (node == 3) {
    to_nodes.push_back(0);
  }

  std::vector<int> from_nodes(0);

  invert_comm_map(to_nodes, from_nodes);

  if (node == 0) {
    if (from_nodes.size() != 3u)
      FAILMSG("Incorrect map size on node 0");
    for (int i = 0; i < 3; ++i) {
      if (from_nodes[i] != i + 1)
        FAILMSG("Incorrent map contents on node 0");
    }
  } else {
    if (from_nodes.size() != 1u)
      FAILMSG("Incorrect map size.");
    if (from_nodes[0] != 0)
      FAILMSG("Incorrect map contents.");
  }

  if (ut.numFails == 0)
    ut.passes("test4 passes");
  else
    ut.failure("test4 failed");

  return;
}

//----------------------------------------------------------------------------//
void test_n_to_n(rtt_c4::ParallelUnitTest &ut) {

  const int nodes = rtt_c4::nodes();

  std::vector<int> to_nodes;
  for (int i = 0; i < nodes; ++i)
    to_nodes.push_back(i);

  std::vector<int> from_nodes;
  invert_comm_map(to_nodes, from_nodes);

  if (static_cast<int>(from_nodes.size()) != nodes)
    FAILMSG("Incorrect from_nodes size.");

  for (int i = 0; i < nodes; ++i) {
    if (to_nodes[i] != from_nodes[i])
      FAILMSG("Incorrect data in map.");
  }

  if (ut.numFails == 0)
    ut.passes("test_n_to_n passes");
  else
    ut.failure("test_n_to_n failed");

  return;
}

//----------------------------------------------------------------------------//
void test_cyclic(rtt_c4::ParallelUnitTest &ut) {

  const int node = rtt_c4::node();
  const int nodes = rtt_c4::nodes();

  std::vector<int> to_nodes(1);
  to_nodes[0] = (node + 1) % nodes;

  std::vector<int> from_nodes;
  invert_comm_map(to_nodes, from_nodes);

  if (from_nodes.size() != 1u)
    FAILMSG("Incorrect map size.");
  if (from_nodes[0] != (node + nodes - 1) % nodes)
    FAILMSG("Incorrect map contents in cyclc test.");

  if (ut.numFails == 0)
    ut.passes("test_cyclic passes");
  else
    ut.failure("test_cyclic failed");

  return;
}

//----------------------------------------------------------------------------//
void test_empty(rtt_c4::ParallelUnitTest &ut) {

  std::vector<int> to_nodes;
  std::vector<int> from_nodes;

  invert_comm_map(to_nodes, from_nodes);

  if (from_nodes.size() != 0u)
    FAILMSG("Incorrect map size in empty test.");

  if (ut.numFails == 0)
    ut.passes("test_empty passes");
  else
    ut.failure("test_empty failed");

  return;
}

//---------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
  rtt_c4::ParallelUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    if (nodes() == 2)
      test2(ut);
    if (nodes() == 4)
      test4(ut);
    test_n_to_n(ut);
    test_cyclic(ut);
    test_empty(ut);
    if (ut.numFails == 0)
      ut.passes("All passed");
    else
      ut.failure("Failed");
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
// end of tstInvert_Comm_Map.cc
//---------------------------------------------------------------------------//
