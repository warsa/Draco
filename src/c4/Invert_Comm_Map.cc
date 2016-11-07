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
/**
 * \brief Specialized version of invert_comm_map for std::vector<int> which
 * avoids data copy operations.
 */
template <> void
invert_comm_map<std::vector<int> >(std::vector<int> const &to_values,
                                   std::vector<int> &from_values) {
    const int myproc = rtt_c4::node();
    const int numprocs = rtt_c4::nodes();
    // flag value to indicate a proc will be writing to this proc.
    const int flag = 1;

    // Create the RMA memory window, which is an array that is numprocs long.
    // Processors that are sending info to this proc will sent their
    // respective value to flag.
    MPI_Win win;
    std::vector<int> proc_flag(numprocs, 0); // initially, all zero
    MPI_Win_create(&proc_flag[0], numprocs * sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Set the remote flags
    MPI_Win_fence(0, win);
    for (int i = 0; i < to_values.size(); ++i)
    {
        Require(to_values[i] >= 0);
        Require(to_values[i] < numprocs);
        if (to_values[i] == myproc)
            proc_flag[myproc] = 1; // ... and our own flag
        else
            MPI_Put(&flag, 1, MPI_INT, to_values[i], myproc, 1, MPI_INT, win);
    }
    MPI_Win_fence(0, win);

    // Back out the from_values from the full flags vector
    from_values.clear();
    for (int i = 0; i < numprocs; ++i)
    {
        Check(proc_flag[i] == 0 || proc_flag[i] == flag);
        if (proc_flag[i] == flag)
            from_values.push_back(i);
    }

    MPI_Win_free(&win);
    return;
}

} // end namespace rtt_c4

//---------------------------------------------------------------------------//
// end of Invert_Comm_Map.cc
//---------------------------------------------------------------------------//
