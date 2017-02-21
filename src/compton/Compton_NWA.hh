//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   compton/Compton_NWA.hh
 * \author Kendra Keady
 * \date   Mon Apr  2 14:14:29 2001
 * \brief  Header file for compton NWA interface -- linked against library
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#ifndef __compton_Compton_NWA_hh__
#define __compton_Compton_NWA_hh__

// C++ standard library dependencies
#include <iostream>
#include <memory>
#include <vector>
// headers provided in compton NWA include directory
#include "etemp_interp.hh"
#include "multigroup_compton_data.hh"

namespace rtt_compton {

class Compton_NWA {

private:
  //! Shared pointer to an electron interpolation object:
  std::shared_ptr<etemp_interp> ei;

public:
  //! Constructor for an existing library
  Compton_NWA(const std::string &);

  Compton_NWA(const std::string &, const std::vector<double> &, const size_t);

  //! Interpolation of all data to a certain electron temperature:
  std::vector<std::vector<std::vector<double>>> interpolate(const double);

  //! Retrieve group structure for the given library:
  std::vector<double> get_group_bounds() {
    return ei->get_Cdata()->get_group_bds();
  }

  //! Retrieve min electron temperature for the given library:
  double get_min_etemp() { return ei->get_min_etemp(); }

  //! Retrieve max electron temperature for the given library:
  double get_max_etemp() { return ei->get_max_etemp(); }

  //! Retrieve number of groups in the given multigroup structure:
  size_t get_num_groups() { return ei->get_Cdata()->get_n_grps(); }

  //! Retrieve number of angular moments/evaluation points in the lib data:
  size_t get_num_xi() { return ei->get_Cdata()->get_n_xi_pts(); }

  //! Retrieve electron temperature eval points (diagnostic use)
  std::vector<double> get_etemp_pts() { return ei->get_etemp_pts(); }
};
}

#endif
