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
#include <vector>

namespace rtt_c4 {

//---------------------------------------------------------------------------//
/**
 * \brief Invert the contents of a one-to-many mapping between nodes.
 *
 * \param[in]  to_values A vector of node numbers that this node communicates
 *             with.
 * \param[out] from_values On output, the vector of node numbers that correspond
 *             to \a to_values.
 *
 * So if the argument \a to_values contains "send to" node values, then the
 * result \a from_values contains the "receive from" node values.  But this
 * routine can also be used as the argument \a to_values contains "receive from"
 * node values, then the result \a from_values contains the "send to" node
 * values.
 */
DLL_PUBLIC_c4 void invert_comm_map(std::vector<int> const &to_values,
                                   std::vector<int> &from_values);

} // end namespace rtt_c4

#endif // c4_Invert_Comm_Map_hh

//---------------------------------------------------------------------------//
// end of c4/Invert_Comm_Map.hh
//---------------------------------------------------------------------------//
