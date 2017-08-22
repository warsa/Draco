//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/test/tstsend_is.cc
 * \author Kelly Thompson
 * \date   Friday, Dec 07, 2012, 14:02 pm
 * \brief  Unit tests for rtt_c4::send_is()
 * \note   Copyright (C) 2016-2017 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//
// $Id: tstsend_is.cc 5830 2011-05-05 19:43:43Z kellyt $
//---------------------------------------------------------------------------//

#include "c4/ParallelUnitTest.hh"
#include "ds++/Release.hh"
#include "ds++/Soft_Equivalence.hh"
#include <sstream>

using rtt_dsxx::soft_equiv;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

// This is a simple class that has a static MPI type and a method to commit that
// type
class Custom {

public:
  Custom(int rank) {
    my_ints[0] = rank;
    my_ints[1] = rank * 1000;
    my_ints[2] = rank * 10000;
    my_doubles[0] = double(rank);
    my_doubles[1] = double(rank * 1000);
    my_longs[0] = rank + 1000000000000;
    my_longs[1] = rank + 10000000000000;
  }
  ~Custom() {}

public:
  static MPI_Datatype MPI_Type;
  static const int mpi_tag = 512;
  static void commit_mpi_type(void) {
    MPI_Datatype og_MPI_Custom;

    const int custom_entry_count = 3;

    // set the number of entries for each datatype
    int num_int(4);
    int num_double(2);
    int num_long(2);
    int custom_array_of_block_length[3] = {num_int, num_double, num_long};

    // Displacements of each type in the cell
    MPI_Aint custom_array_of_block_displace[3] = {
        0, num_int * sizeof(int),
        num_int * sizeof(int) + num_double * sizeof(double)};

    //Type of each memory block
    MPI_Datatype custom_array_of_types[3] = {MPI_INT, MPI_DOUBLE, MPI_LONG};

    MPI_Type_create_struct(custom_entry_count, custom_array_of_block_length,
                           custom_array_of_block_displace,
                           custom_array_of_types, &og_MPI_Custom);

    // Commit the type to MPI so it recognizes it in communication calls
    MPI_Type_commit(&og_MPI_Custom);

    // Duplicate the type so it's recognized when returned out of this
    // context (I don't know why this is necessary)
    MPI_Type_dup(og_MPI_Custom, &MPI_Type);
  }

  int get_int1(void) const { return my_ints[0]; }
  int get_int2(void) const { return my_ints[1]; }
  int get_int3(void) const { return my_ints[3]; }
  double get_double1(void) const { return my_doubles[0]; }
  double get_double2(void) const { return my_doubles[1]; }
  long get_long1(void) const { return my_longs[0]; }
  long get_long2(void) const { return my_longs[1]; }

private:
  int my_ints[3];
  double my_doubles[2];
  long my_longs[2];
};

// the static data member needs to be defined outside the class
MPI_Datatype Custom::MPI_Type = MPI_Datatype();

void test_simple(rtt_dsxx::UnitTest &ut) {
  // borrowed from http://mpi.deino.net/mpi_functions/MPI_Issend.html.

  if (rtt_c4::node() == 0)
    std::cout << "Test send_is() by sending data to proc myid+1..."
              << std::endl;

  // C4_Req communication handles.
  std::vector<rtt_c4::C4_Req> comm_int(2);

  // for point-to-point communiction we need to know neighbor's identifiers:
  // left, right.
  int right = (rtt_c4::node() + 1) % rtt_c4::nodes();
  int left = rtt_c4::node() - 1;
  if (left < 0)
    left = rtt_c4::nodes() - 1;

  // create some data to send/recv
  unsigned int const bsize(10);
  std::vector<int> buffer2(bsize);
  std::vector<int> buffer1(bsize);
  for (size_t i = 0; i < bsize; ++i)
    buffer1[i] = 1000 * rtt_c4::node() + i;

  // post asynchronous receives.
  comm_int[0] = rtt_c4::receive_async(&buffer2[0], buffer2.size(), left);

  try {
    // send data using non-blocking synchronous send.
    rtt_c4::send_is(comm_int[1], &buffer1[0], buffer1.size(), right);

    // wait for all communication to finish
    rtt_c4::wait_all(comm_int.size(), &comm_int[0]);

    // exected results
    std::vector<int> expected(bsize);
    for (size_t i = 0; i < bsize; ++i)
      expected[i] = 1000 * left + i;

    if (expected == buffer2) {
      std::ostringstream msg;
      msg << "Expected data found after send_is() on node " << rtt_c4::node()
          << ".";
      PASSMSG(msg.str());
    } else {
      std::ostringstream msg;
      msg << "Did not find expected data after send_is() on node "
          << rtt_c4::node() << ".";
      FAILMSG(msg.str());
    }
  } catch (rtt_dsxx::assertion const &error) {
#ifdef C4_SCALAR
    PASSMSG("Successfully caught a ds++ exception while trying to use "
            "send_is() in a C4_SCALAR build.");
#else
    FAILMSG("Encountered a ds++ exception while testing send_is().");
#endif
  }

  return;
}

void test_send_custom(rtt_dsxx::UnitTest &ut) {
  // borrowed from http://mpi.deino.net/mpi_functions/MPI_Issend.html.

  // commit the MPI type for the Custom class. This must be done before
  // send_is_custom is called. DMC checks will throw if the type has not been
  // committed because size comparison will fail and MPI throws an error when
  // an uncommited type is used in a send/receive
  Custom::commit_mpi_type();

  if (rtt_c4::node() == 0) {
    std::cout << "Test send_is_custom() by sending data to proc myid+1..."
              << std::endl;
    int custom_mpi_type_size(0);
    MPI_Type_size(Custom::MPI_Type, &custom_mpi_type_size);
    std::cout << " Size of custom type: " << sizeof(Custom) << std::endl;
    std::cout << " Size of custom MPI type: " << custom_mpi_type_size
              << std::endl;
  }

  // C4_Req communication handles.
  std::vector<rtt_c4::C4_Req> comm_int(2);

  // for point-to-point communiction we need to know neighbor's identifiers:
  // left, right.
  int right = (rtt_c4::node() + 1) % rtt_c4::nodes();
  int left = rtt_c4::node() - 1;
  if (left < 0)
    left = rtt_c4::nodes() - 1;

  // create some data to send/recv
  Custom my_custom_object(rtt_c4::node());

  // post asynchronous receives.
  Custom recv_custom_object(-1);
  rtt_c4::receive_async_custom(comm_int[0], &recv_custom_object, 1, left,
                               Custom::mpi_tag, Custom::MPI_Type);

  try {
    // send data using non-blocking synchronous send. Custom sends check to make\
    // sure that the type
    rtt_c4::send_is_custom(comm_int[1], &my_custom_object, 1, right,
                           Custom::mpi_tag, Custom::MPI_Type);

    // wait for all communication to finish
    rtt_c4::wait_all(comm_int.size(), &comm_int[0]);

    // check that the exected results match the custom type from the left rank
    Custom expected_custom(left);

    if (expected_custom.get_int1() != recv_custom_object.get_int1())
      ITFAILS;
    if (expected_custom.get_int2() != recv_custom_object.get_int2())
      ITFAILS;
    if (expected_custom.get_int3() != recv_custom_object.get_int3())
      ITFAILS;
    if (!soft_equiv(expected_custom.get_double1(),
                    recv_custom_object.get_double1()))
      ITFAILS;
    if (!soft_equiv(expected_custom.get_double2(),
                    recv_custom_object.get_double2()))
      ITFAILS;
    if (expected_custom.get_long1() != recv_custom_object.get_long1())
      ITFAILS;
    if (expected_custom.get_long2() != recv_custom_object.get_long2())
      ITFAILS;

  } catch (rtt_dsxx::assertion const &error) {
#ifdef C4_SCALAR
    PASSMSG("Successfully caught a ds++ exception while trying to use "
            "send_is_custom() in a C4_SCALAR build.");
#else
    FAILMSG("Encountered a ds++ exception while testing send_is_custom().");
#endif
  }

  return;
}

//---------------------------------------------------------------------------//
int main(int argc, char *argv[]) {
  rtt_c4::ParallelUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    test_simple(ut);
    test_send_custom(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
// end of tstsend_is.cc
//---------------------------------------------------------------------------//
