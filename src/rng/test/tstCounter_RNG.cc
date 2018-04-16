//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   rng/test/tstCounter_RNG.cc
 * \author Peter Ahrens
 * \date   Fri Aug 3 16:53:23 2012
 * \brief  Counter_RNG tests.
 * \note   Copyright (C) 2016-2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "ds++/Release.hh"
#include "ds++/ScalarUnitTest.hh"
#include "ds++/Soft_Equivalence.hh"
#include "rng/Counter_RNG.hh"
#include <set>

using namespace std;
using namespace rtt_dsxx;
using namespace rtt_rng;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

void test_equality(UnitTest &ut) {
  // Create a Counter_RNG by specifying a seed and stream number.
  uint32_t seed = 1;
  uint64_t streamnum = 2;
  Counter_RNG rng(seed, streamnum);

  if (rng.get_num() != streamnum)
    ITFAILS;
  if (rng.size() != CBRNG_DATA_SIZE)
    ITFAILS;
  if (rng.size_bytes() != CBRNG_DATA_SIZE * sizeof(uint64_t))
    ITFAILS;
  if (rng != rng)
    ITFAILS;

  // Create another Counter_RNG with a different seed.
  seed = 2;
  Counter_RNG rng2(seed, streamnum);

  // rng2's stream number should match rng's, but the two generators should not
  // be identical.
  if (rng2.get_num() != streamnum)
    ITFAILS;
  if (rng2.get_num() != rng.get_num())
    ITFAILS;
  if (rng2.get_unique_num() == rng.get_unique_num())
    ITFAILS;
  if (rng2 == rng)
    ITFAILS;
  if (rng2 != rng2)
    ITFAILS;

  // Create another Counter_RNG with a different stream number.
  seed = 1;
  streamnum = 3;
  Counter_RNG rng3(seed, streamnum);

  // rng3 should be different from the previous two generators.
  if (rng3.get_num() != streamnum)
    ITFAILS;
  if (rng3.get_unique_num() == rng.get_unique_num())
    ITFAILS;
  if (rng3.get_unique_num() == rng2.get_unique_num())
    ITFAILS;
  if (rng3 == rng)
    ITFAILS;
  if (rng3 == rng2)
    ITFAILS;
  if (rng3 != rng3)
    ITFAILS;

  // Create another Counter_RNG with the original seed and stream number.
  streamnum = 2;
  Counter_RNG rng4(seed, streamnum);

  // rng4 should be equal to rng but different from rng2 and rng3.
  if (rng4.get_num() != streamnum)
    ITFAILS;
  if (rng4.get_unique_num() != rng.get_unique_num())
    ITFAILS;
  if (rng4.get_unique_num() == rng2.get_unique_num())
    ITFAILS;
  if (rng4.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (rng4 != rng)
    ITFAILS;
  if (rng4 == rng2)
    ITFAILS;
  if (rng4 == rng3)
    ITFAILS;
  if (rng4 != rng4)
    ITFAILS;

  // Create a Counter_RNG from a data array.
  vector<uint64_t> data(CBRNG_DATA_SIZE);
  data[0] = 1234;
  data[1] = 5678;
  data[2] = 9012;
  data[3] = 3456;
  Counter_RNG rng5(&data[0], &data[0] + CBRNG_DATA_SIZE);

  streamnum = data[2];
  if (rng5.get_num() != streamnum)
    ITFAILS;
  if (rng5.get_unique_num() == rng.get_unique_num())
    ITFAILS;
  if (rng5.get_unique_num() == rng2.get_unique_num())
    ITFAILS;
  if (rng5.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (rng5.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (rng5 == rng)
    ITFAILS;
  if (rng5 == rng2)
    ITFAILS;
  if (rng5 == rng3)
    ITFAILS;
  if (rng5 == rng4)
    ITFAILS;
  if (rng5 != rng5)
    ITFAILS;

  // Create a Counter_RNG from a data array that should match rng and rng4.
  data[0] = 0;
  data[1] = static_cast<uint64_t>(1) << 32;
  data[2] = 2;
  data[3] = 0;
  Counter_RNG rng6(&data[0], &data[0] + CBRNG_DATA_SIZE);

  streamnum = data[2];
  if (rng6.get_num() != streamnum)
    ITFAILS;
  if (rng6.get_unique_num() != rng.get_unique_num())
    ITFAILS;
  if (rng6.get_unique_num() == rng2.get_unique_num())
    ITFAILS;
  if (rng6.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (rng6.get_unique_num() != rng4.get_unique_num())
    ITFAILS;
  if (rng6.get_unique_num() == rng5.get_unique_num())
    ITFAILS;
  if (rng6 != rng)
    ITFAILS;
  if (rng6 == rng2)
    ITFAILS;
  if (rng6 == rng3)
    ITFAILS;
  if (rng6 != rng4)
    ITFAILS;
  if (rng6 == rng5)
    ITFAILS;
  if (rng6 != rng6)
    ITFAILS;

// Try to create a Counter_RNG from a data array that's too short.
// 1. Only test exceptions if DbC is enabled.
// 2. However, do not run these tests if no-throw DbC is enabled (DBC & 8)
#ifdef REQUIRE_ON
#if !(DBC & 8)
  bool caught = false;
  try {
    Counter_RNG rng7(&data[0], &data[0] + CBRNG_DATA_SIZE - 1);
  } catch (rtt_dsxx::assertion &err) {
    cout << "Good, caught assertion: " << err.what() << endl;
    caught = true;
  }
  if (!caught)
    ITFAILS;
#endif
#endif

  // Test for equality using iterators.
  if (!std::equal(rng6.begin(), rng6.end(), rng.begin()))
    ITFAILS;
  if (std::equal(rng6.begin(), rng6.end(), rng2.begin()))
    ITFAILS;
  if (std::equal(rng6.begin(), rng6.end(), rng3.begin()))
    ITFAILS;
  if (!std::equal(rng6.begin(), rng6.end(), rng4.begin()))
    ITFAILS;
  if (std::equal(rng6.begin(), rng6.end(), rng5.begin()))
    ITFAILS;

  if (ut.numFails == 0)
    PASSMSG("test_equality passed");
}

//---------------------------------------------------------------------------//
void test_stream(UnitTest &ut) {
  // Create two identical Counter_RNGs.
  uint32_t seed = 0x12121212;
  uint64_t streamnum = 1234;
  Counter_RNG rng(seed, streamnum);
  Counter_RNG rng2(seed, streamnum);

  if (rng != rng2)
    ITFAILS;

  // Generate a random double (and advance the stream) from rng.
  double x = rng.ran();

  // rng and rng2 should no longer match, but their stream numbers and unique
  // identifiers should be the same.
  if (rng == rng2)
    ITFAILS;
  if (rng.get_num() != streamnum)
    ITFAILS;
  if (rng.get_num() != rng2.get_num())
    ITFAILS;
  if (rng.get_unique_num() != rng2.get_unique_num())
    ITFAILS;

  // Generate a random double (and advance the stream) from rng2.
  double y = rng2.ran();

  // Now rng and rng2 should match again, and the two generated doubles should
  // be identical.
  if (rng != rng2)
    ITFAILS;
  if (!soft_equiv(x, y))
    ITFAILS;

  // Generate another random double from rng.
  double z = rng.ran();

  // Now they should differ again.
  if (rng == rng2)
    ITFAILS;
  if (rng.get_num() != streamnum)
    ITFAILS;
  if (rng.get_num() != rng2.get_num())
    ITFAILS;
  if (rng.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (soft_equiv(x, z))
    ITFAILS;

  // Create a Counter_RNG from a data array.
  vector<uint64_t> data(CBRNG_DATA_SIZE);
  data[0] = 0;
  data[1] = static_cast<uint64_t>(seed) << 32;
  data[2] = streamnum;
  data[3] = 0;
  Counter_RNG rng3(&data[0], &data[0] + CBRNG_DATA_SIZE);

  // Initially, rng3 should exactly match neither rng nor rng2, but all three
  // should have the same stream number and "unique" identifier.
  if (!std::equal(rng3.begin(), rng3.end(), data.begin()))
    ITFAILS;
  if (rng3 == rng)
    ITFAILS;
  if (rng3 == rng2)
    ITFAILS;
  if (rng3.get_num() != streamnum)
    ITFAILS;
  if (rng3.get_unique_num() != rng.get_unique_num())
    ITFAILS;
  if (rng3.get_unique_num() != rng2.get_unique_num())
    ITFAILS;

  // Generate a random double from rng3; it should match rng2 but not data
  // afterward.
  double w = rng3.ran();
  if (rng3 == rng)
    ITFAILS;
  if (rng3 != rng2)
    ITFAILS;
  if (std::equal(rng3.begin(), rng3.end(), data.begin()))
    ITFAILS;
  if (rng3.get_num() != streamnum)
    ITFAILS;
  if (rng3.get_unique_num() != rng.get_unique_num())
    ITFAILS;
  if (rng3.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (!soft_equiv(w, y))
    ITFAILS;

  if (ut.numFails == 0)
    PASSMSG("test_stream passed");
}

//---------------------------------------------------------------------------//
void test_alias(UnitTest &ut) {
  // Create four Counter_RNGs; rng and rng2 are identical, and rng, rng2, and
  // rng3 have the same stream number.
  uint64_t streamnum = 0x20202020;
  Counter_RNG rng(0x1111, streamnum);
  Counter_RNG rng2(0x1111, streamnum);
  Counter_RNG rng3(0x2222, streamnum);
  ++streamnum;
  Counter_RNG rng4(0x3333, streamnum);

  if (rng.get_num() != rng2.get_num())
    ITFAILS;
  if (rng.get_num() != rng3.get_num())
    ITFAILS;
  if (rng.get_num() == rng4.get_num())
    ITFAILS;
  if (rng2.get_num() != rng3.get_num())
    ITFAILS;
  if (rng2.get_num() == rng4.get_num())
    ITFAILS;
  if (rng3.get_num() == rng4.get_num())
    ITFAILS;
  if (rng.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (rng.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (rng.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (rng2.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (rng2.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (rng3.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (rng != rng2)
    ITFAILS;
  if (rng == rng3)
    ITFAILS;
  if (rng == rng4)
    ITFAILS;
  if (rng2 == rng3)
    ITFAILS;
  if (rng2 == rng4)
    ITFAILS;
  if (rng3 == rng4)
    ITFAILS;

  // Create a Counter_RNG_Ref from rng.
  Counter_RNG_Ref ref(rng.ref());

  if (ref.get_num() != rng.get_num())
    ITFAILS;
  if (ref.get_num() != rng2.get_num())
    ITFAILS;
  if (ref.get_num() != rng3.get_num())
    ITFAILS;
  if (ref.get_num() == rng4.get_num())
    ITFAILS;
  if (ref.get_unique_num() != rng.get_unique_num())
    ITFAILS;
  if (ref.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (ref.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (ref.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (!ref.is_alias_for(rng))
    ITFAILS;
  if (ref.is_alias_for(rng2))
    ITFAILS;
  if (ref.is_alias_for(rng3))
    ITFAILS;
  if (ref.is_alias_for(rng4))
    ITFAILS;

  // Generate a random double (and advance the stream) from rng via ref.
  double x = ref.ran();

  if (ref.get_num() != rng.get_num())
    ITFAILS;
  if (ref.get_num() != rng2.get_num())
    ITFAILS;
  if (ref.get_num() != rng3.get_num())
    ITFAILS;
  if (ref.get_num() == rng4.get_num())
    ITFAILS;
  if (ref.get_unique_num() != rng.get_unique_num())
    ITFAILS;
  if (ref.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (ref.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (ref.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (!ref.is_alias_for(rng))
    ITFAILS;
  if (ref.is_alias_for(rng2))
    ITFAILS;
  if (ref.is_alias_for(rng3))
    ITFAILS;
  if (ref.is_alias_for(rng4))
    ITFAILS;

  // Invoking ref.ran should have altered rng; it should still have the same
  // stream number as rng2 and rng3, but it should be identical to none of them.
  if (rng.get_num() != rng2.get_num())
    ITFAILS;
  if (rng.get_num() != rng3.get_num())
    ITFAILS;
  if (rng.get_num() == rng4.get_num())
    ITFAILS;
  if (rng.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (rng.get_unique_num() == rng3.get_unique_num())
    ITFAILS;
  if (rng.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (rng == rng2)
    ITFAILS;
  if (rng == rng3)
    ITFAILS;
  if (rng == rng4)
    ITFAILS;

  // Create a bare data array that should match rng.
  vector<uint64_t> data(CBRNG_DATA_SIZE);
  data[0] = 1;
  data[1] = static_cast<uint64_t>(0x1111) << 32;
  data[2] = 0x20202020;
  data[3] = 0;

  if (!std::equal(rng.begin(), rng.end(), data.begin()))
    ITFAILS;

  // Create a Counter_RNG_Ref from a bare data array.
  data[0] = 0;
  data[1] = static_cast<uint64_t>(0x2222) << 32;
  data[2] = 0x20202020;
  data[3] = 0;
  Counter_RNG_Ref ref2(&data[0], &data[0] + CBRNG_DATA_SIZE);

  // ref2 should have the same stream number as rng, rng2, and rng3 but
  // shouldn't be an alias for any of them.
  if (ref2.get_num() != rng.get_num())
    ITFAILS;
  if (ref2.get_num() != rng2.get_num())
    ITFAILS;
  if (ref2.get_num() != rng3.get_num())
    ITFAILS;
  if (ref2.get_num() == rng4.get_num())
    ITFAILS;
  if (ref2.get_unique_num() == rng.get_unique_num())
    ITFAILS;
  if (ref2.get_unique_num() == rng2.get_unique_num())
    ITFAILS;
  if (ref2.get_unique_num() != rng3.get_unique_num())
    ITFAILS;
  if (ref2.get_unique_num() == rng4.get_unique_num())
    ITFAILS;
  if (ref2.is_alias_for(rng))
    ITFAILS;
  if (ref2.is_alias_for(rng2))
    ITFAILS;
  if (ref2.is_alias_for(rng3))
    ITFAILS;
  if (ref2.is_alias_for(rng4))
    ITFAILS;

  // Generate a random double from ref.
  double y = ref2.ran();

  // The underlying data array should have changed.
  if (data[0] != 1)
    ITFAILS;
  if (data[1] != static_cast<uint64_t>(0x2222) << 32)
    ITFAILS;
  if (data[2] != 0x20202020)
    ITFAILS;
  if (data[3] != 0)
    ITFAILS;
  if (soft_equiv(y, x))
    ITFAILS;

  // Generate a random double from rng3; it should match the one from ref2.
  double z = rng3.ran();

  if (!soft_equiv(z, y))
    ITFAILS;
  if (soft_equiv(z, x))
    ITFAILS;

// Try to create a Counter_RNG_Ref with a data array that's too short.
// 1. Only test exceptions if DbC is enabled.
// 2. However, do not run these tests if no-throw DbC is enabled (DBC & 8)
#ifdef REQUIRE_ON
#if !(DBC & 8)
  bool caught = false;
  try {
    Counter_RNG_Ref ref3(&data[0], &data[0] + CBRNG_DATA_SIZE - 1);
  } catch (rtt_dsxx::assertion &err) {
    cout << "Good, caught assertion: " << err.what() << endl;
    caught = true;
  }
  if (!caught)
    ITFAILS;
#endif
#endif

  if (ut.numFails == 0)
    PASSMSG("test_alias passed");
}

//---------------------------------------------------------------------------//
void test_rollover(UnitTest &ut) {
  // Create a Counter_RNG with a large counter value.
  vector<uint64_t> data(CBRNG_DATA_SIZE);
  data[0] = 0xfffffffffffffffd;
  data[1] = 1;
  data[2] = 0xabcd;
  data[3] = 0xef00;
  Counter_RNG rng(&data[0], &data[0] + CBRNG_DATA_SIZE);

  // Increment data[0], generate a random double, and compare.
  ++data[0];
  double x = rng.ran();
  if (!std::equal(rng.begin(), rng.end(), data.begin()))
    ITFAILS;

  // ... and again.
  ++data[0];
  double y = rng.ran();
  if (soft_equiv(y, x))
    ITFAILS;
  if (!std::equal(rng.begin(), rng.end(), data.begin()))
    ITFAILS;

  // Generate another random double and verify that the counter has incremented
  // correctly.
  data[0] = 0;
  data[1] = 2;
  double z = rng.ran();
  if (soft_equiv(z, x))
    ITFAILS;
  if (soft_equiv(z, y))
    ITFAILS;
  if (!std::equal(rng.begin(), rng.end(), data.begin()))
    ITFAILS;

  // Repeat the test with a Counter_RNG_Ref.
  data[0] = 0xfffffffffffffffe;
  data[1] = 1;
  Counter_RNG_Ref ref(&data[0], &data[0] + CBRNG_DATA_SIZE);

  double y2 = ref.ran();
  if (!soft_equiv(y2, y))
    ITFAILS;
  if (data[0] != 0xffffffffffffffff)
    ITFAILS;
  if (data[1] != 1)
    ITFAILS;
  if (data[2] != 0xabcd)
    ITFAILS;
  if (data[3] != 0xef00)
    ITFAILS;

  double z2 = ref.ran();
  if (!soft_equiv(z2, z))
    ITFAILS;
  if (data[0] != 0)
    ITFAILS;
  if (data[1] != 2)
    ITFAILS;
  if (data[2] != 0xabcd)
    ITFAILS;
  if (data[3] != 0xef00)
    ITFAILS;

  if (ut.numFails == 0)
    PASSMSG("test_rollover passed");
}

//---------------------------------------------------------------------------//
void test_unique(UnitTest &ut) {
  // Create three identical generators.
  uint32_t seed = 332211;
  uint64_t streamnum = 2468;
  Counter_RNG rng(seed, streamnum);
  Counter_RNG rng2(seed, streamnum);
  Counter_RNG rng3(seed, streamnum);

  Counter_RNG_Ref rng_ref(rng.ref());
  Counter_RNG_Ref rng2_ref(rng2.ref());
  Counter_RNG_Ref rng3_ref(rng3.ref());

  if (rng != rng2)
    ITFAILS;
  if (rng != rng3)
    ITFAILS;
  if (rng.get_unique_num() != rng2.get_unique_num())
    ITFAILS;
  if (rng.get_unique_num() != rng3.get_unique_num())
    ITFAILS;

  if (!rng_ref.is_alias_for(rng))
    ITFAILS;
  if (!rng2_ref.is_alias_for(rng2))
    ITFAILS;
  if (!rng3_ref.is_alias_for(rng3))
    ITFAILS;

  // Generate some random numbers from rng2.  The stream number and unique
  // number of rng2 should remain the same during this process.
  set<uint64_t> ids;
  ids.insert(rng.get_unique_num());

  for (int i = 0; i < 1000000; ++i) {
    rng2.ran();

    if (rng2.get_num() != rng.get_num())
      ITFAILS;
    if (rng2_ref.get_unique_num() != rng2.get_unique_num())
      ITFAILS;
    if (ids.find(rng2.get_unique_num()) == ids.end())
      ITFAILS;
  }

  if (ut.numFails == 0)
    PASSMSG("test_unique passed");
}

//---------------------------------------------------------------------------//

int main(int argc, char *argv[]) {
  ScalarUnitTest ut(argc, argv, release);
  try {
    test_equality(ut);
    test_stream(ut);
    test_alias(ut);
    test_rollover(ut);
    test_unique(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
// end of tstCounter_RNG.cc
//---------------------------------------------------------------------------//
