//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   compton/Compton_NWA.hh
 * \author Kendra Keady
 * \date   Tues Feb 21 2017
 * \brief  Implementation file for compton NWA interface
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved.
 */

// headers provided in draco:
#include "Compton_NWA.hh"
#include "ds++/Assert.hh"
// headers provided in Compton_NWA include directory:
#include "compton_file.hh"
#include "multigroup_compton_data.hh"
#include "multigroup_data_types.hh"
#include "multigroup_lib_builder.hh"

namespace rtt_compton {

// ------------ //
// Constructors //
// ------------ //

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor for an existing multigroup libfile.
 *
 * This calls NWA methods to read the data file and store everything in a
 * Compton data object, a smart pointer to which is then passed to (and held by)
 * the NWA etemp_interp class.
 *
 * \param filehandle The name of the multigroup file to use for Compton scatters
 */
Compton_NWA::Compton_NWA(const std::string &filehandle) {

  // Make a compton file object
  compton_file Cfile(false);

  // read the (existing multigroup) library into a compton data container
  std::shared_ptr<multigroup_compton_data> Cdata =
      Cfile.read_mg_csk_data(filehandle);

  // be sure the Cdata pointer is non-null:
  Ensure(Cdata);

  // initialize the electron temperature interpolator with the mg compton data
  ei.reset(new etemp_interp(Cdata));

  // Make sure the SP exists...
  Ensure(ei);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor for an existing pointwise file and a multigroup structure.
 *
 * This calls NWA methods to read the pointwise library and construct a
 * multigroup Compton data object, a smart pointer to which is then passed
 * to (and held by) the NWA etemp_interp class.
 *
 * \param filehandle The name of the pointwise lib to build MG data from
 * \param grp_bds    A vector containing the multigroup bounds (in keV)
 * \param n_xi       The number of angular points/Legendre moments desired
 */
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

  // Make an mg lib builder, based on the group structure and pointwise lib:
  multigroup_lib_builder MG_builder(grp_data);

  // build the library:
  MG_builder.build_library();

  // package the data into a compton_data pointer:
  std::shared_ptr<multigroup_compton_data> Cdata = MG_builder.package_data();

  // Make sure the SP exists...
  Ensure(Cdata);

  // initialize the electron temperature interpolator with the mg compton data
  ei.reset(new etemp_interp(Cdata));

  // Make sure the SP exists...
  Ensure(ei);
}

// ------------ //
//  Interfaces  //
// ------------ //

//---------------------------------------------------------------------------//
/*!
 * \brief Interpolated data to a given SCALED electron temperature (T / m_e)
 *
 * This method interpolates MG Compton lib data to a given electron temperature.
 * It returns the interpolated values for ALL g, g', and angular points
 * in the specified multigroup structure.
 *
 * \param etemp The SCALED electron temperature ( temp / electron rest-mass )
 * \return      n_grp x n_grp x n_xi interpolated scattering kernel values
 */
std::vector<std::vector<std::vector<double>>>
Compton_NWA::interpolate(const double etemp) {

  // Be sure the passed electron temperature is within the bounds of the lib!
  Require(etemp >= ei->get_min_etemp());
  Require(etemp <= ei->get_max_etemp());

  // return value
  std::vector<std::vector<std::vector<double>>> interped_data;

  // call the appropriate routine in the electron interp object
  // (defined in Compton_NWA lib)
  interped_data = ei->interpolate_etemp(etemp);

  return interped_data;
}
}
