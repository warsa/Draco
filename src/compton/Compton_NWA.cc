//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   compton/Compton_NWA.hh
 * \author Kendra Keady
 * \date   Mon Apr  2 14:14:29 2001
 * \brief  Implementation file for compton NWA interface -- linked against library
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#include "Compton_NWA.hh"
#include "compton_file.hh"
#include "multigroup_compton_data.hh"
#include "multigroup_data_types.hh"
#include "multigroup_lib_builder.hh"
#include "ds++/Assert.hh"

namespace rtt_compton {

// Constructor for an existing multigroup libfile.
// This calls NWA methods to read the data file and store everything in a
// Compton data object, a smart pointer to which is then passed to (and held by)
// the NWA etemp_interp class.
Compton_NWA::Compton_NWA(const std::string &filehandle) {

  // Make a compton file object
  compton_file Cfile(false);

  std::shared_ptr<multigroup_compton_data> Cdata =
      Cfile.read_mg_csk_data(filehandle);

  Ensure(Cdata);

  ei.reset(new etemp_interp(Cdata));

  Ensure(ei);
}

// Constructor for an existing pointwise file and a multigroup structure.
// This calls NWA methods to read the pointwise library and construct a
// multigroup Compton data object, a smart pointer to which is then passed
// to (and held by) the NWA etemp_interp class.
Compton_NWA::Compton_NWA(const std::string &filehandle,
                         const std::vector<double> &grp_bds, const size_t nxi) {

  // make a group_data struct to pass to the lib builder:
  multigroup::Group_data grp_data;
  grp_data.group_bounds = grp_bds;
  grp_data.lib_file = filehandle;
  grp_data.n_leg = nxi;

  // This is totally arbitrary, because I'm just assuming the user wants
  // Planck-weighting...
  grp_data.wt_func = multigroup::Weighting_function::PLANCK;

  // TODO: How do we actually want to handle the weighting function?
  // Allow the user to pass it in? Make some intelligent decision at runtime?

  std::cout << "*********************************************************\n"
            << "WARNING! Building a multigroup library from scratch might\n"
            << " take a LOOOOOOONG time! (Don't say I didn't warn you.)  \n"
            << "*********************************************************\n"
            << std::endl;

  multigroup_lib_builder MG_builder(grp_data);

  MG_builder.build_library();

  std::shared_ptr<multigroup_compton_data> Cdata = MG_builder.package_data();

  Ensure(Cdata);

  ei.reset(new etemp_interp(Cdata));

  Ensure(ei);
}

// This method interpolates data to a given SCALED electron temperature, etemp
// It returns the interpolated values for each g, g', and angle/Legendre moment.
std::vector<std::vector<std::vector<double>>>
Compton_NWA::interpolate(const double etemp) {

  // Be sure the passed electron temperature is within the bounds of the lib!
  Require(etemp >= ei->get_min_etemp());
  Require(etemp <= ei->get_max_etemp());

  std::vector<std::vector<std::vector<double>>> interped_data;

  interped_data = ei->interpolate_etemp(etemp);

  return interped_data;
}
}
