//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/Invert_Comm_Map.cc
 * \author Mike Buksas, Rob Lowrie
 * \date   Mon Nov 19 10:09:11 2007
 * \brief  Implementation of Invert_Comm_Map
 * \note   Copyright (C) 2016 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//

#include "Invert_Comm_Map.hh"
#include "ds++/Assert.hh"

namespace rtt_c4 {

//---------------------------------------------------------------------------//
// MPI version of invert_comm_map
#ifdef C4_MPI
void invert_comm_map(std::vector<int> const &to_values,
                     std::vector<int> &from_values) {
  const int myproc = rtt_c4::node();
  const int numprocs = rtt_c4::nodes();

  // value to indicate a proc will be communicating with myproc.
  int flag = 1;

  // The vector that the other procs will set the flag value, if they
  // are writing to the current proc.
  std::vector<int> proc_flag(numprocs, 0); // initially, all zero

  // Create the RMA memory window of the vector.
  MPI_Win win;
  MPI_Win_create(&proc_flag[0], numprocs * sizeof(int), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  // Assertion value for fences.  Currently, we effectively don't set
  // anything (zero).
  const int fence_assert = 0;

  // Set the local and remote vector values
  MPI_Win_fence(fence_assert, win);
  for (auto it = to_values.begin(); it != to_values.end(); ++it) {
    Require(*it >= 0);
    Require(*it < numprocs);
    if (*it == myproc) {
      // ... set our local value
      proc_flag[myproc] = 1;
    } else {
      // ... set the value on the remote proc
      MPI_Put(&flag, 1, MPI_INT, *it, myproc, 1, MPI_INT, win);
    }
  }
  MPI_Win_fence(fence_assert, win);

  // Back out the from_values from the full flags vector
  from_values.clear();
  for (int i = 0; i < numprocs; ++i) {
    Check(proc_flag[i] == 0 || proc_flag[i] == flag);
    if (proc_flag[i] == flag)
      from_values.push_back(i);
  }

  MPI_Win_free(&win);
  return;
}
//---------------------------------------------------------------------------//
// SCALAR version of invert_comm_map
#elif defined(C4_SCALAR)
void invert_comm_map(std::vector<int> const &to_values,
                     std::vector<int> &from_values) {
  Require(to_values.size() <= 1);
  from_values.clear();
  if (to_values.size() > 0 && to_values[0] == 0) {
    from_values.push_back(0);
  }
}
#else
//---------------------------------------------------------------------------//
// Default version of invert_comm_map, which throws an error.
void invert_comm_map(std::vector<int> const &, std::vector<int> &) {
  Insist(0, "invert_comm_map not implemented for this communication type!");
}
#endif // ifdef C4_MPI

} // end namespace rtt_c4

//---------------------------------------------------------------------------//
// end of Invert_Comm_Map.cc
//---------------------------------------------------------------------------//
