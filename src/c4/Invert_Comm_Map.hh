//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/Invert_Comm_Map.hh
 * \author Mike Buksas, Rob Lowrie
 * \date   Mon Nov 19 10:09:10 2007
 * \brief  Implementation of Invert_Comm_Map
 * \note   Copyright (C) 2016 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//

#ifndef c4_Invert_Comm_Map_hh
#define c4_Invert_Comm_Map_hh

#include "C4_Functions.hh"
#include <iterator> // back_inserter
#include <vector>

namespace rtt_c4 {

//---------------------------------------------------------------------------//
/**
 * \brief Invert the contents of a one-to-many mapping between nodes.
 *
 * The argument on each node is a container whose stored values are other
 * nodes. The result is collection of the same type containing the nodes which
 * referred to the local node.
 *
 * E.g. if the argument contains "communicate to" node values. The result
 * contains the "receive from" node values.
 */
template <typename T> void invert_comm_map(const T &to_values, T &from_values) {

    // Copy the provided container to a std::vector<int>
    std::vector<int> to_data;
    std::vector<int> from_data;
    to_data.insert(to_data.end(), to_values.begin(), to_values.end());
    
    invert_comm_map<std::vector<int> >(to_data, from_data);
    
    // Append the results to the end of the provided container.
    std::copy(from_data.begin(), from_data.end(),
              std::back_inserter(from_values));
    return;
}

// Specialization.  See implementation file.
template <> void
invert_comm_map<std::vector<int> >(std::vector<int> const &to_values,
                                   std::vector<int> &from_values);

} // end namespace rtt_c4

#endif // c4_Invert_Comm_Map_hh

//---------------------------------------------------------------------------//
// end of c4/Invert_Comm_Map.hh
//---------------------------------------------------------------------------//
