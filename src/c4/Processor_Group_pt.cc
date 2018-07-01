//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/Processor_Group_pt.hh
 * \author Kent Budge
 * \date   Fri Oct 20 13:49:10 2006
 * \brief  Member definitions of class Processor_Group
 * \note   Copyright (C) 2016-2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "c4/config.h"

#ifdef C4_MPI

#include "Processor_Group.t.hh"

namespace rtt_c4 {

template DLL_PUBLIC_c4 void Processor_Group::sum(std::vector<double> &values);

template void DLL_PUBLIC_c4 Processor_Group::assemble_vector(
    std::vector<double> const &local, std::vector<double> &global) const;

template void DLL_PUBLIC_c4 Processor_Group::assemble_vector(
    double const *local, double *global, unsigned count) const;

} // end namespace rtt_c4

#endif //C4_MPI

//---------------------------------------------------------------------------//
// end of c4/Processor_Group_pt.hh
//---------------------------------------------------------------------------//
