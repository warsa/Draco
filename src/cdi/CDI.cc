//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   cdi/CDI.cc
 * \author Kelly Thompson
 * \date   Thu Jun 22 16:22:07 2000
 * \brief  CDI class implementation file.
 * \note   Copyright (C) 2016-2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "CDI.hh"
#include "ds++/Safe_Divide.hh"
#include "ds++/Soft_Equivalence.hh"
#include <iostream>
#include <limits>
#include <numeric>

namespace rtt_cdi {

//---------------------------------------------------------------------------//
// CONSTRUCTORS AND DESTRUCTORS
//---------------------------------------------------------------------------//

/*!
 * \brief Construct a CDI object.
 *
 * Builds a CDI object.  The opacity and eos objects that this holds must be
 * loaded using the set functions.  There is no easy way to guarantee that all
 * of the set objects point to the same material.  CDI does do checking that
 * only one of each Model:Reaction pair of opacity objects are assigned;
 * however, the user can "fake" CDI with different materials if he/she is
 * malicious enough.
 *
 * CDI does allow a string material ID indicator.  It is up to the client to
 * ascribe meaning to the indicator.
 *
 * \param[in] id string material id descriptor, this is defaulted to null
 */
CDI::CDI(const std_string &id)
    : grayOpacities(constants::num_Models,
                    SF_GrayOpacity(constants::num_Reactions)),
      multigroupOpacities(constants::num_Models,
                          SF_MultigroupOpacity(constants::num_Reactions)),
      odfmgOpacities(constants::num_Models,
                     SF_OdfmgOpacity(constants::num_Reactions)),
      spEoS(SP_EoS()), spEICoupling(SP_EICoupling()), matID(id) {
  Ensure(grayOpacities.size() == constants::num_Models);
  Ensure(multigroupOpacities.size() == constants::num_Models);
  Ensure(odfmgOpacities.size() == constants::num_Models);
}

//---------------------------------------------------------------------------//

CDI::~CDI() { /* empty */
}

//---------------------------------------------------------------------------//
// STATIC DATA
//---------------------------------------------------------------------------//

std::vector<double> CDI::frequencyGroupBoundaries = std::vector<double>();
std::vector<double> CDI::opacityCdfBandBoundaries = std::vector<double>();

//---------------------------------------------------------------------------//
// STATIC FUNCTIONS
//---------------------------------------------------------------------------//

/*!
 * \brief Return the frequency group boundaries.
 *
 * Every multigroup opacity object held by any CDI object contains the same
 * frequency group boundaries.  This static function allows CDI users to access
 * the group boundaries without referencing a particular material.
 *
 * Note, the group boundaries are not set until a multigroup opacity object is
 * set for the first time (in any CDI object) with the setMultigroupOpacity
 * function.
 */
std::vector<double> CDI::getFrequencyGroupBoundaries() {
  return frequencyGroupBoundaries;
}
/*!
 * \brief Return the opacity band boundaries.
 *
 * Every multiband opacity object held by any CDI object contains the same band
 * boundaries, and also inside each group.  This static function allows CDI
 * users to access the band boundaries without referencing a particular
 * material.
 *
 * Note, the band boundaries are not set until a multigroup opacity object is
 * set for the first time (in any CDI object) with the setOdfmgOpacity function.
 */
std::vector<double> CDI::getOpacityCdfBandBoundaries() {
  return opacityCdfBandBoundaries;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return the number of frequency groups.
 */
size_t CDI::getNumberFrequencyGroups() {
  return frequencyGroupBoundaries.empty() ? 0
                                          : frequencyGroupBoundaries.size() - 1;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return the number of opacity bands.
 */
size_t CDI::getNumberOpacityBands() {
  return opacityCdfBandBoundaries.empty() ? 0
                                          : opacityCdfBandBoundaries.size() - 1;
}

//---------------------------------------------------------------------------//
// Core Integrators
/*
 * These are the most basic of the Planckian and Rosseland integration
 * functions. They are used in the implementation of integration functions with
 * friendlier interfaces.
 */
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Planckian Spectrum Integrators
//
/* These are versions of the integrators that work over specific energy ranges
 * or groups in the stored group structure.
 */
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 *
 * \brief Integrate the Planckian spectrum over a frequency group.
 *
 * \param[in] groupIndex Index of the frequency group to integrate
 *                       [1,num_groups].
 * \param[in] T          The temperature in keV (must be greater than 0.0).
 *
 * \return Integrated normalized Plankian over the group specified by
 *         groupIndex.
 */
double CDI::integratePlanckSpectrum(size_t const groupIndex, double const T) {
  Insist(!frequencyGroupBoundaries.empty(), "No groups defined!");

  Require(T >= 0.0);
  Require(groupIndex > 0);
  Require(groupIndex <= frequencyGroupBoundaries.size() - 1);

  // Determine the group boundaries for groupIndex
  const double lower_bound = frequencyGroupBoundaries[groupIndex - 1];
  const double upper_bound = frequencyGroupBoundaries[groupIndex];
  Check(upper_bound > lower_bound);

  double integral = integratePlanckSpectrum(lower_bound, upper_bound, T);

  Ensure(integral >= 0.0);
  Ensure(integral <= 1.0);

  return integral;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Integrate the Planckian spectrum over all frequency groups.
 * \param[in] T The temperature in keV (must be greater than 0.0).
 * \return Integrated normalized Plankian over all frequency groups.
 */
double CDI::integratePlanckSpectrum(const double T) {
  Insist(!frequencyGroupBoundaries.empty(), "No groups defined!");
  Require(T >= 0.0);

  // first determine the group boundaries for groupIndex
  const double lower_bound = frequencyGroupBoundaries.front();
  const double upper_bound = frequencyGroupBoundaries.back();
  Check(upper_bound > lower_bound);

  // calculate the integral
  double integral = integratePlanckSpectrum(lower_bound, upper_bound, T);

  Ensure(integral >= 0.0 && integral <= 1.0);

  return integral;
}

//---------------------------------------------------------------------------//
// Rosseland Spectrum Integrators
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 *
 * \brief Integrate the Rosseland spectrum over a frequency group.
 *
 * \param[in] groupIndex index of the frequency group to integrate
 *                       [1,num_groups]
 * \param[in] T          the temperature in keV (must be greater than 0.0)
 * \return integrated normalized Plankian over the group specified by
 *         groupIndex.
 */
double CDI::integrateRosselandSpectrum(size_t const groupIndex,
                                       double const T) {
  Insist(!frequencyGroupBoundaries.empty(), "No groups defined!");
  Require(T >= 0.0);
  Require(groupIndex > 0 && groupIndex <= frequencyGroupBoundaries.size() - 1);

  // first determine the group boundaries for groupIndex
  const double lowFreq = frequencyGroupBoundaries[groupIndex - 1];
  const double highFreq = frequencyGroupBoundaries[groupIndex];
  Check(highFreq > lowFreq);

  double rosseland = integrateRosselandSpectrum(lowFreq, highFreq, T);

  return rosseland;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Integrate the Planckian and Rosseland spectrum over a frequency group.
 *
 * \param groupIndex index of the frequency group to integrate [1,num_groups]
 * \param T          The temperature in keV (must be greater than 0.0)
 * \param planck     Reference argument for the Planckian integral
 * \param rosseland  Reference argument for the Rosseland integral
 *
 * \return The integrated normalized Planckian and Rosseland over the requested
 *         frequency group. These are returned as references in argument planck and
 *         rosseland
 */
void CDI::integrate_Rosseland_Planckian_Spectrum(const size_t groupIndex,
                                                 const double T, double &planck,
                                                 double &rosseland) {
  Insist(!frequencyGroupBoundaries.empty(), "No groups defined!");

  Require(T >= 0.0);
  Require(groupIndex > 0);
  Require(groupIndex <= frequencyGroupBoundaries.size() - 1);

  // Determine the group boundaries
  const double lowFreq = frequencyGroupBoundaries[groupIndex - 1];
  const double highFreq = frequencyGroupBoundaries[groupIndex];
  Check(highFreq > lowFreq);

  // Call the general frequency version
  integrate_Rosseland_Planckian_Spectrum(lowFreq, highFreq, T, planck,
                                         rosseland);
  return;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Integrate the Planckian Specrum over an entire a set of frequency
 *        groups, returning a vector of the integrals
 *
 * \param bounds The vector of group boundaries. Size n+1
 * \param T The temperature
 * \param planck Return argument containing the Planckian integrals. Size n
 */
void CDI::integrate_Planckian_Spectrum(std::vector<double> const &bounds,
                                       double const T,
                                       std::vector<double> &planck) {
  Require(T >= 0.0);
  Require(bounds.size() > 0);

  size_t const groups(bounds.size() - 1);
  planck.resize(groups, 0.0);

  // nu/T must be < numeric_limits<double>::max().  So, if T < nu*min(), then
  // return early with planck == 0.0. This avoids a possible divide by zero.
  if (T <= bounds[0] * std::numeric_limits<double>::min())
    return;

  // Initialize the loop:
  double scaled_frequency = bounds[0] / T;
  double planck_value = integrate_planck(scaled_frequency);

  for (size_t group = 0; group < groups; ++group) {
    // Shift the data down:
    Remember(double const last_scaled_frequency = scaled_frequency;);
    double const last_planck = planck_value;

    // New values:
    if (T <= bounds[group + 1] * std::numeric_limits<double>::min())
      planck[group] = 0.0;
    else {
      scaled_frequency = bounds[group + 1] / T;
      Ensure(scaled_frequency > last_scaled_frequency);
      planck_value = integrate_planck(scaled_frequency);

      // Record the definite integral between frequencies.
      planck[group] = planck_value - last_planck;
    }
    Ensure(planck[group] >= 0.0);
    Ensure(planck[group] <= 1.0);
  }
  return;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Integrate the Planckian Specrum over an entire a set of frequency
 *        groups directly returning the vector of the integrals
 *
 * \param bounds The vector of group boundaries. Size n+1
 * \param T The temperature
 * \return vector containing the Planckian integrals. Size n_groups
 */
std::vector<double>
CDI::integrate_Planckian_Spectrum(std::vector<double> const &bounds,
                                  double const T) {
  Require(T >= 0.0);
  Require(bounds.size() > 0);

  size_t const groups(bounds.size() - 1);
  std::vector<double> planck(groups, 0.0);

  // nu/T must be < numeric_limits<double>::max().  So, if T < nu*min(), then
  // return early with planck == 0.0. This avoids a possible divide by zero.
  if (T <= bounds[0] * std::numeric_limits<double>::min())
    return planck;

  // Initialize the loop:
  double scaled_frequency = bounds[0] / T;
  double planck_value = integrate_planck(scaled_frequency);

  for (size_t group = 0; group < groups; ++group) {
    // Shift the data down:
    Remember(double const last_scaled_frequency = scaled_frequency;);
    double const last_planck = planck_value;

    // New values:
    if (T <= bounds[group + 1] * std::numeric_limits<double>::min())
      planck[group] = 0.0;
    else {
      scaled_frequency = bounds[group + 1] / T;
      Ensure(scaled_frequency > last_scaled_frequency);
      planck_value = integrate_planck(scaled_frequency);

      // Record the definite integral between frequencies.
      planck[group] = planck_value - last_planck;
    }
    Ensure(planck[group] >= 0.0);
    Ensure(planck[group] <= 1.0);
  }
  return planck;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Integrate the Rosseland Spectrum over an entire a set of frequency
 *        groups, returning a vector of the integrals
 *
 * \param bounds    The vector of group boundaries (size n+1)
 * \param T         The temperature
 * \param rosseland Return argument containing the Rosseland integrals (size n)
 */
void CDI::integrate_Rosseland_Spectrum(std::vector<double> const &bounds,
                                       double const T,
                                       std::vector<double> &rosseland) {
  Require(T >= 0.0);
  Require(bounds.size() > 0);

  size_t const groups(bounds.size() - 1);
  rosseland.resize(groups, 0.0);

  // nu/T must be < numeric_limits<double>::max().  So, if T < nu*min(), then
  // return early with rosseland == 0.0. This avoids a possible divide by zero.
  if (T <= bounds[0] * std::numeric_limits<double>::min())
    return;

  // Initialize the loop:
  double scaled_frequency = bounds[0] / T;
  double exp_scaled_frequency = std::exp(-scaled_frequency);

  double planck_value(-42.0);
  double rosseland_value(-42.0);
  integrate_planck_rosseland(scaled_frequency, exp_scaled_frequency,
                             planck_value, rosseland_value);

  for (size_t group = 0; group < groups; ++group) {

    // Shift the data down:
    Remember(double const last_scaled_frequency = scaled_frequency;);
    double const last_rosseland = rosseland_value;

    // New values:
    if (T <= bounds[group + 1] * std::numeric_limits<double>::min())
      rosseland[group] = 0.0;
    else {
      scaled_frequency = bounds[group + 1] / T;
      Check(scaled_frequency > last_scaled_frequency);
      exp_scaled_frequency = std::exp(-scaled_frequency);
      integrate_planck_rosseland(scaled_frequency, exp_scaled_frequency,
                                 planck_value, rosseland_value);

      // Record the definite integral between frequencies.
      rosseland[group] = rosseland_value - last_rosseland;
    }
    Ensure(rosseland[group] >= 0.0);
    Ensure(rosseland[group] <= 1.0);
  }
  return;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Integrate the Planckian and Rosseland Specrum over an entire a set of
 *        frequency groups, returning a vector of the integrals
 *
 * \param bounds    The vector of group boundaries. Size n+1
 * \param T         The temperature
 * \param planck    Return argument containing the Planckian integrals. Size n
 * \param rosseland Return argumant containing the Rosseland integrals. Size n
 */
void CDI::integrate_Rosseland_Planckian_Spectrum(
    std::vector<double> const &bounds, double const T,
    std::vector<double> &planck, std::vector<double> &rosseland) {

  Require(T >= 0.0);
  Require(bounds.size() > 0);
  size_t const groups(bounds.size() - 1);

  planck.resize(groups, 0.0);
  rosseland.resize(groups, 0.0);

  // nu/T must be < numeric_limits<double>::max().  So, if T < nu*min(), then
  // return early with planck ==0.0 and rosseland == 0.0. This avoids a possible
  // divide by zero
  if (T <= bounds[0] * std::numeric_limits<double>::min())
    return;

  // Initialize the loop:
  double scaled_frequency = bounds[0] / T;
  double exp_scaled_frequency = std::exp(-scaled_frequency);

  double planck_value(-42.0);
  double rosseland_value(-42.0);
  integrate_planck_rosseland(scaled_frequency, exp_scaled_frequency,
                             planck_value, rosseland_value);

  for (size_t group = 0; group < groups; ++group) {

    // Shift the data down:
    Remember(double const last_scaled_frequency = scaled_frequency;);
    double const last_planck = planck_value;
    double const last_rosseland = rosseland_value;

    // New values:
    if (T <= bounds[group + 1] * std::numeric_limits<double>::min()) {
      planck[group] = 0.0;
      rosseland[group] = 0.0;
    } else {

      scaled_frequency = bounds[group + 1] / T;
      Check(scaled_frequency > last_scaled_frequency);
      exp_scaled_frequency = std::exp(-scaled_frequency);
      integrate_planck_rosseland(scaled_frequency, exp_scaled_frequency,
                                 planck_value, rosseland_value);

      // Record the definite integral between frequencies.
      planck[group] = planck_value - last_planck;
      rosseland[group] = rosseland_value - last_rosseland;
    }
    Ensure(planck[group] >= 0.0);
    Ensure(planck[group] <= 1.0);
    Ensure(rosseland[group] >= 0.0);
    Ensure(rosseland[group] <= 1.0);
  }
  return;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Collapse a multigroup opacity set into a single representative value
 *        weighted by the Planckian function.
 *
 * \param groupBounds The vector of group boundaries.
 * \param opacity   A vector of multigroup opacity data.
 * \param planckSpectrum A vector of Planck integrals for all groups in the
 *                  spectrum (normally generated via
 *                  CDI::integrate_Rosseland_Planckian_Sectrum(...).
 * \param emission_group_cdf
 * \return A single interval Planckian weighted opacity value.
 *
 * Typically, CDI::integrate_Rosseland_Planckian_Spectrum is called before this
 * function to obtain planckSpectrum.
 */
double CDI::collapseMultigroupOpacitiesPlanck(
    std::vector<double> const &groupBounds, std::vector<double> const &opacity,
    std::vector<double> const &planckSpectrum,
    std::vector<double> &emission_group_cdf) {
  Require(groupBounds.size() > 0);
  Require(opacity.size() == groupBounds.size() - 1);
  Require(planckSpectrum.size() == groupBounds.size() - 1);
  Require(emission_group_cdf.size() == groupBounds.size() - 1);

  // Integrate the unnormalized Planckian over the group spectrum
  // int_{\nu_0}^{\nu_G}{d\nu B(\nu,T)}
  double const planck_integral =
      std::accumulate(planckSpectrum.begin(), planckSpectrum.end(), 0.0);
  Check(planck_integral >= 0.0);

  // Perform integration of sigma * b_g over all groups:
  // int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}

  // Initialize sum:
  double sig_planck_sum(0.0);
  // Multiply by the absorption opacity and accumulate.
  for (size_t g = 1; g < groupBounds.size(); ++g) {
    Check(planckSpectrum[g - 1] >= 0.0);
    Check(opacity[g - 1] >= 0.0);
    sig_planck_sum += planckSpectrum[g - 1] * opacity[g - 1];
    // Also collect some CDF data.
    emission_group_cdf[g - 1] = sig_planck_sum;
  }

  //                         int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}
  // Planck opac:  sigma_P = --------------------------------------------------
  //                         int_{\nu_0}^{\nu_G}{d\nu * B(\nu,T)}

  double planck_opacity(0.0);
  if (planck_integral > 0.0)
    planck_opacity = sig_planck_sum / planck_integral;
  else {
    // Weak check that the zero integrated Planck is due to a cold temperature
    // whose Planckian peak is below the lowest (first) group boundary.
    Check(rtt_dsxx::soft_equiv(sig_planck_sum, 0.0));
    // Check( T >= 0.0 );
    // Check( 3.0 * T <= groupBounds[0] );

    // Set the ill-defined integrated Planck opacity to zero.
    // planck_opacity = 0.0; // already initialized to zero.
  }
  Ensure(planck_opacity >= 0.0);
  return planck_opacity;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Collapse a multigroup opacity set into a single representative value
 *        weighted by the Planckian function without setting the emission CDF
 *
 * \param groupBounds The vector of group boundaries.
 * \param opacity   A vector of multigroup opacity data.
 * \param planckSpectrum A vector of Planck integrals for all groups in the
 *                  spectrum (normally generated via
 *                  CDI::integrate_Rosseland_Planckian_Sectrum(...).
 * \return A single interval Planckian weighted opacity value.
 *
 * Typically, CDI::integrate_Rosseland_Planckian_Spectrum is called before this
 * function to obtain planckSpectrum.
 */
double CDI::collapseMultigroupOpacitiesPlanck(
    std::vector<double> const &groupBounds, std::vector<double> const &opacity,
    std::vector<double> const &planckSpectrum) {
  Require(groupBounds.size() > 0);
  Require(opacity.size() == groupBounds.size() - 1);
  Require(planckSpectrum.size() == groupBounds.size() - 1);

  // Integrate the unnormalized Planckian over the group spectrum
  // int_{\nu_0}^{\nu_G}{d\nu B(\nu,T)}
  double const planck_integral =
      std::accumulate(planckSpectrum.begin(), planckSpectrum.end(), 0.0);
  Check(planck_integral >= 0.0);

  // Perform integration of sigma * b_g over all groups:
  // int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}

  // Initialize sum:
  double sig_planck_sum(0.0);
  // Multiply by the absorption opacity and accumulate.
  for (size_t g = 1; g < groupBounds.size(); ++g) {
    Check(planckSpectrum[g - 1] >= 0.0);
    Check(opacity[g - 1] >= 0.0);
    sig_planck_sum += planckSpectrum[g - 1] * opacity[g - 1];
  }

  //                         int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}
  // Planck opac:  sigma_P = --------------------------------------------------
  //                         int_{\nu_0}^{\nu_G}{d\nu * B(\nu,T)}

  double planck_opacity(0.0);
  if (planck_integral > 0.0)
    planck_opacity = sig_planck_sum / planck_integral;
  else {
    // Weak check that the zero integrated Planck is due to a cold temperature
    // whose Planckian peak is below the lowest (first) group boundary.
    Check(rtt_dsxx::soft_equiv(sig_planck_sum, 0.0));
    // Check( T >= 0.0 );
    // Check( 3.0 * T <= groupBounds[0] );

    // Set the ill-defined integrated Planck opacity to zero.
    // planck_opacity = 0.0; // already initialized to zero.
  }
  Ensure(planck_opacity >= 0.0);
  return planck_opacity;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Collapse a multigroup reciprocal opacity set into a single
 *        representative value weighted by the Planckian function.
 *
 * \param groupBounds The vector of group boundaries.
 * \param opacity   A vector of multigroup opacity data.
 * \param planckSpectrum A vector of Planck integrals for all groups in the
 *                  spectrum (normally generated via
 *                  CDI::integrate_Rosseland_Planckian_Sectrum(...).
 * \return A single interval Planckian weighted reciprocal opacity value.
 *
 * Typically, CDI::integrate_Rosseland_Planckian_Spectrum is called before this
 * function to obtain planckSpectrum.
 */
double CDI::collapseMultigroupReciprocalOpacitiesPlanck(
    std::vector<double> const &groupBounds, std::vector<double> const &opacity,
    std::vector<double> const &planckSpectrum) {
  Require(groupBounds.size() > 0);
  Require(opacity.size() == groupBounds.size() - 1);
  Require(planckSpectrum.size() == groupBounds.size() - 1);

  // Integrate the unnormalized Planckian over the group spectrum
  // int_{\nu_0}^{\nu_G}{d\nu B(\nu,T)}
  double const planck_integral =
      std::accumulate(planckSpectrum.begin(), planckSpectrum.end(), 0.0);
  Check(planck_integral >= 0.0);

  // Perform integration of sigma * b_g over all groups:
  // int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}

  // Initialize sum:
  double inv_sig_planck_sum(0.0);
  // Multiply by the absorption opacity and accumulate.
  for (size_t g = 1; g < groupBounds.size(); ++g) {
    Check(planckSpectrum[g - 1] >= 0.0);
    Check(opacity[g - 1] >= 0.0);
    if (opacity[g - 1] > 0)
      inv_sig_planck_sum += planckSpectrum[g - 1] / opacity[g - 1];
    else
      return std::numeric_limits<float>::max();
  }

  //                             int_{\nu_0}^{\nu_G}{d\nu 1/sigma(\nu,T) * B(\nu,T)}
  // Planck opac:  inv_sigma_P = --------------------------------------------------
  //                              int_{\nu_0}^{\nu_G}{d\nu * B(\nu,T)}

  double reciprocal_planck_opacity(0.0);
  if (planck_integral > 0.0)
    reciprocal_planck_opacity = inv_sig_planck_sum / planck_integral;
  else {
    reciprocal_planck_opacity = std::numeric_limits<float>::max();
  }
  Ensure(reciprocal_planck_opacity >= 0.0);
  return reciprocal_planck_opacity;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Collapse a multigroup opacity set into a single representative value
 *        weighted by the Rosseland function.
 *
 * \param groupBounds The vector of group boundaries. Size n+1
 * \param opacity   A vector of multigroup opacity data.
 * \param rosselandSpectrum A vector of Rosseland integrals for all groups in
 *                  the spectrum (normally generated via
 *                  CDI::integrate_Rosseland_Planckian_Sectrum(...).
 * \return A single interval Rosseland weighted opacity value.
 *
 * Typically, CDI::integrate_Rosseland_Planckian_Spectrum is called before this
 * function to obtain rosselandSpectrum.
 *
 * There are 2 special cases that we check for:
 * 1. All opacities are zero - just return 0.0;
 * 2. The Rosseland Integral is very small (or zero).  In this case, perform a
 *    modified calculation that does not depend on the Rosseland integral.
 *
 * If neither of the special cases are in effect, then do the normal evaluation.
 */
double CDI::collapseMultigroupOpacitiesRosseland(
    std::vector<double> const &groupBounds, std::vector<double> const &opacity,
    std::vector<double> const &rosselandSpectrum) {
  Require(groupBounds.size() > 0);
  Require(opacity.size() == groupBounds.size() - 1);
  Require(rosselandSpectrum.size() == groupBounds.size() - 1);

  // If all opacities are zero, then the Rosseland mean will also be zero.
  double const eps(std::numeric_limits<double>::epsilon());
  double const opacity_sum =
      std::accumulate(opacity.begin(), opacity.end(), 0.0);
  if (rtt_dsxx::soft_equiv(opacity_sum, 0.0, eps)) {
    // std::cerr << "\nWARNING ("
    //           << "CDI.cc::collapseMultigroupOpacitiesRosseland)::"
    //           << "\n\tComputing Rosseland Opacity when all opacities "
    //           << "are zero!" << std::endl;
    return 0.0;
  }

  // Integrate the unnormalized Rosseland over the group spectrum
  // int_{\nu_0}^{\nu_G}{d\nu dB(\nu,T)/dT}
  double const rosseland_integral =
      std::accumulate(rosselandSpectrum.begin(), rosselandSpectrum.end(), 0.0);

  // If the group bounds are well outside the Rosseland Spectrum at the current
  // temperature, our algorithm may return a value that is within machine
  // precision of zero.  In this case, we assume that this occurs when the
  // temperature -> 0, so that limit(T->0) dB/dT = \delta(\nu).  In this case we
  // have:
  //
  // sigma_R = sigma(g=0)

  if (rosseland_integral < eps)
    return opacity[0];

  // Analytically the Rosseland integral should always be > 0.
  Check(rosseland_integral > 0.0);

  // Perform integration of (1/sigma) * d(b_g)/dT over all groups:
  // int_{\nu_0}^{\nu_G}{d\nu (1/sigma(\nu,T)) * dB(\nu,T)/dT}

  // Rosseland opacity:

  //    1      int_{\nu_0}^{\nu_G}{d\nu (1/sigma(\nu,T)) * dB(\nu,T)/dT}
  // ------- = ----------------------------------------------------------
  // sigma_R   int_{\nu_0}^{\nu_G}{d\nu dB(\nu,T)/dT}

  // Initialize sum
  double inv_sig_r_sum(0.0);

  // Accumulated quantities for the Rosseland opacities:
  for (size_t g = 1; g < groupBounds.size(); ++g) {
    if (rosselandSpectrum[g - 1] / rosseland_integral > eps) {
      inv_sig_r_sum +=
          rtt_dsxx::safe_pos_divide(rosselandSpectrum[g - 1], opacity[g - 1]);
    }
  }
  Check(inv_sig_r_sum > 0.0);
  return rosseland_integral / inv_sig_r_sum;
}
//---------------------------------------------------------------------------//
/*!
 * \brief Collapse a multigroup-multiband opacity set into a single
 *        representative value weighted by the Planckian function.
 *
 * \param groupBounds The vector of group boundaries.
 * \param opacity   A vector of multigroup opacity data.
 * \param planckSpectrum A vector of Planck integrals for all groups in the
 *                  spectrum (normally generated via
 *                  CDI::integrate_Rosseland_Planckian_Sectrum(...).
 * \param bandWidths Vector of energy band widths
 * \param emission_group_cdf
 * \return A single interval Planckian weighted opacity value.
 *
 * Typically, CDI::integrate_Rosseland_Planckian_Spectrum is called before this
 * function to obtain planckSpectrum.
 */
double CDI::collapseOdfmgOpacitiesPlanck(
    std::vector<double> const &groupBounds,
    std::vector<std::vector<double>> const &opacity,
    std::vector<double> const &planckSpectrum,
    std::vector<double> const &bandWidths,
    std::vector<std::vector<double>> &emission_group_cdf) {
  Require(groupBounds.size() > 0);
  Require(opacity.size() == groupBounds.size() - 1);
  Require(opacity[0].size() == bandWidths.size());
  Require(planckSpectrum.size() == groupBounds.size() - 1);
  Require(emission_group_cdf.size() == groupBounds.size() - 1);
  Require(emission_group_cdf[0].size() == bandWidths.size());

  // Integrate the unnormalized Planckian over the group spectrum
  // int_{\nu_0}^{\nu_G}{d\nu B(\nu,T)}
  double const planck_integral =
      std::accumulate(planckSpectrum.begin(), planckSpectrum.end(), 0.0);
  Check(planck_integral >= 0.0);

  // Perform integration of sigma * b_g over all groups:
  // int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}

  // Initialize sum:
  double sig_planck_sum(0.0);

  size_t const numGroups = groupBounds.size() - 1;
  size_t const numBands = bandWidths.size();

  // Multiply by the absorption opacity and accumulate.
  for (size_t g = 1; g <= numGroups; ++g) {
    for (size_t ib = 1; ib <= numBands; ++ib) {
      Check(planckSpectrum[g - 1] >= 0.0);
      Check(opacity[g - 1][ib - 1] >= 0.0);

      sig_planck_sum +=
          planckSpectrum[g - 1] * bandWidths[ib - 1] * opacity[g - 1][ib - 1];
      // Also collect some CDF data.
      emission_group_cdf[g - 1][ib - 1] = sig_planck_sum;
    }
  }

  //                         int_{\nu_0}^{\nu_G}{d\nu sigma(\nu,T) * B(\nu,T)}
  // Planck opac:  sigma_P = --------------------------------------------------
  //                         int_{\nu_0}^{\nu_G}{d\nu * B(\nu,T)}

  double planck_opacity(0.0);
  // if( planck_integral > 0.0 )
  planck_opacity = sig_planck_sum / planck_integral;
  // else
  // {
  //     // Weak check that the zero integrated Planck is due to a cold
  //     // temperature whose Planckian peak is below the lowest (first)
  //     // group boundary.
  //     Check( rtt_dsxx::soft_equiv(sig_planck_sum, 0.0) );
  //     Check( T >= 0.0 );
  //     Check( 3.0 * T <= groupBounds[0] );

  //     // Set the ill-defined integrated Planck opacity to zero.
  //     // planck_opacity = 0.0; // already initialized to zero.
  // }
  Ensure(planck_opacity >= 0.0);
  return planck_opacity;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Collapse a multigroup-multiband opacity set into a single
 *        representative reciprocal value weighted by the Planckian function.
 *
 * \param groupBounds The vector of group boundaries.
 * \param opacity   A vector of multigroup opacity data.
 * \param planckSpectrum A vector of Planck integrals for all groups in the
 *                  spectrum (normally generated via
 *                  CDI::integrate_Rosseland_Planckian_Sectrum(...).
 * \param bandWidths Vector of energy band widths
 *
 * Typically, CDI::integrate_Rosseland_Planckian_Spectrum is called before this
 * function to obtain planckSpectrum.
 */
double CDI::collapseOdfmgReciprocalOpacitiesPlanck(
    std::vector<double> const &groupBounds,
    std::vector<std::vector<double>> const &opacity,
    std::vector<double> const &planckSpectrum,
    std::vector<double> const &bandWidths) {
  Require(groupBounds.size() > 0);
  Require(opacity.size() == groupBounds.size() - 1);
  Require(opacity[0].size() == bandWidths.size());
  Require(planckSpectrum.size() == groupBounds.size() - 1);

  // Integrate the unnormalized Planckian over the group spectrum
  // int_{\nu_0}^{\nu_G}{d\nu B(\nu,T)}
  double const planck_integral =
      std::accumulate(planckSpectrum.begin(), planckSpectrum.end(), 0.0);
  Check(planck_integral >= 0.0);

  // Perform integration of b_g/sigma over all groups:
  // int_{\nu_0}^{\nu_G}{d\nu B(\nu,T) / sigma(\nu,T)}

  // Initialize sum:
  double inv_sig_planck_sum(0.0);

  size_t const numGroups = groupBounds.size() - 1;
  size_t const numBands = bandWidths.size();

  // Multiply by the absorption opacity and accumulate.
  for (size_t g = 1; g <= numGroups; ++g) {
    for (size_t ib = 1; ib <= numBands; ++ib) {
      Check(planckSpectrum[g - 1] >= 0.0);
      Check(opacity[g - 1][ib - 1] >= 0.0);
      Check((g - 1) * numBands + ib - 1 < numBands * numGroups);
      double denom = opacity[g - 1][ib - 1];
      if (denom > 0)
        inv_sig_planck_sum +=
            planckSpectrum[g - 1] * bandWidths[ib - 1] / denom;
      else
        return std::numeric_limits<float>::max();
    }
  }

  //                             int_{\nu_0}^{\nu_G}{d\nu B(\nu,T) / sigma(\nu,T)}
  // Planck opac:  inv_sigma_P = --------------------------------------------------
  //                             int_{\nu_0}^{\nu_G}{d\nu * B(\nu,T)}

  double reciprocal_planck_opacity(0.0);
  if (planck_integral > 0.0)
    reciprocal_planck_opacity = inv_sig_planck_sum / planck_integral;
  else
    reciprocal_planck_opacity = std::numeric_limits<float>::max();

  Ensure(reciprocal_planck_opacity >= 0.0);
  return reciprocal_planck_opacity;
}

//---------------------------------------------------------------------------//
// SET FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * \brief Register a gray opacity (rtt_cdi::GrayOpacity) with CDI.
 *
 * This function sets a gray opacity object of type rtt_cdi::GrayOpacity with
 * the CDI object.  It stores the gray opacity object based upon its
 * rtt_cdi::Model and rtt_cdi::Reaction types.  If a GrayOpacity with these
 * types has already been registered an exception is thrown.  To register a new
 * set of GrayOpacity objects call CDI::reset() first.  You cannot overwrite
 * registered objects with the setGrayOpacity() function!
 *
 * \param spGOp smart pointer to a GrayOpacity object
 */
void CDI::setGrayOpacity(const SP_GrayOpacity &spGOp) {
  Require(spGOp);

  // determine the model and reaction type (these MUST be in the correct range
  // because the Model and Reaction are constrained by the rtt_cdi::Model and
  // rtt_cdi::Reaction enumerations, assuming nobody hosed these)
  rtt_cdi::Model model = spGOp->getModelType();
  rtt_cdi::Reaction reaction = spGOp->getReactionType();

  Insist(!grayOpacities[model][reaction],
         "Tried to overwrite a set GrayOpacity object!");

  // assign the smart pointer
  grayOpacities[model][reaction] = spGOp;

  Ensure(grayOpacities[model][reaction]);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Register a multigroup opacity (rtt_cdi::MultigroupOpacity) with CDI.
 *
 * This function sets a multigroup opacity object of type
 * rtt_cdi::MultigroupOpacity with the CDI object.  It stores the multigroup
 * opacity object based upon its rtt_cdi::Model and rtt_cdi::Reaction types.  If
 * a MultigroupOpacity with these type has already been registered an exception
 * is thrown.  To register a new set of MultigroupOpacity objects call
 * CDI::reset() first.  You cannot overwrite registered objects with the
 * setMultigroupOpacity() function!
 *
 * \param spMGOp smart pointer to a MultigroupOpacity object
 */
void CDI::setMultigroupOpacity(const SP_MultigroupOpacity &spMGOp) {
  using rtt_dsxx::soft_equiv;

  Require(spMGOp);

  // determine the model and reaction types
  rtt_cdi::Model model = spMGOp->getModelType();
  rtt_cdi::Reaction reaction = spMGOp->getReactionType();

  Insist(!multigroupOpacities[model][reaction],
         "Tried to overwrite a set MultigroupOpacity object!");

  // if the frequency group boundaries have not been assigned in any CDI object,
  // then assign them here
  if (frequencyGroupBoundaries.empty()) {
    // copy the the group boundaries for this material to the "global" group
    // boundaries that will be enforced for all CDI objects
    frequencyGroupBoundaries = spMGOp->getGroupBoundaries();
  }

  // always check that the number of frequency groups is the same for each
  // multigroup material added to CDI
  Insist(spMGOp->getNumGroupBoundaries() == frequencyGroupBoundaries.size(),
         "Incompatible frequency groups assigned for this material");

  // do a check of the actual boundary values when DBC check is on (this is more
  // expensive so we retain the option of turning it off)
  Remember(std::vector<double> const ref = spMGOp->getGroupBoundaries(););
  Check(soft_equiv(frequencyGroupBoundaries.begin(),
                   frequencyGroupBoundaries.end(), ref.begin(), ref.end(),
                   1.0e-6));

  // assign the smart pointer
  multigroupOpacities[model][reaction] = spMGOp;

  Ensure(multigroupOpacities[model][reaction]);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Register a multigroup opacity (rtt_cdi::OdfmgOpacity) with CDI.
 *
 * This function sets a multigroup opacity object of type rtt_cdi::OdfmgOpacity
 * with the CDI object.  It stores the multigroup opacity object based upon its
 * rtt_cdi::Model and rtt_cdi::Reaction types.  If a OdfmgOpacity with these
 * type has already been registered an exception is thrown.  To register a new
 * set of OdfmgOpacity objects call CDI::reset() first.  You cannot overwrite
 * registered objects with the setOdfmgOpacity() function!
 *
 * \param spODFOp smart pointer to a OdfmgOpacity object
 */
void CDI::setOdfmgOpacity(const SP_OdfmgOpacity &spODFOp) {
  using rtt_dsxx::soft_equiv;

  Require(spODFOp);

  // determine the model and reaction types
  rtt_cdi::Model model = spODFOp->getModelType();
  rtt_cdi::Reaction reaction = spODFOp->getReactionType();

  Insist(!odfmgOpacities[model][reaction],
         "Tried to overwrite a set odfmgOpacity object!");

  // if the frequency group boundaries have not been assigned in any CDI object,
  // then assign them here
  if (frequencyGroupBoundaries.empty()) {
    // copy the the group boundaries for this material to the "global" group
    // boundaries that will be enforced for all CDI objects
    frequencyGroupBoundaries = spODFOp->getGroupBoundaries();
  }

  // if the opacity band boundaries have not been assigned in any CDI object,
  // then assign them here
  if (opacityCdfBandBoundaries.empty()) {
    // copy the the band boundaries for this material to the "global" band
    // boundaries that will be enforced for all CDI objects
    opacityCdfBandBoundaries = spODFOp->getBandBoundaries();
  }

  // always check that the number of frequency groups is the same for each odfmg
  // material added to CDI
  Insist(spODFOp->getNumGroupBoundaries() == frequencyGroupBoundaries.size(),
         "Incompatible frequency groups assigned for this material");

  // always check that the number of frequency groups is the same for each odfmg
  // material added to CDI
  Insist(spODFOp->getNumBandBoundaries() == opacityCdfBandBoundaries.size(),
         "Incompatible opacity bands assigned for this material");

  // do a check of the actual boundary values when DBC check is on (this is more
  // expensive so we retain the option of turning it off)
  Remember(std::vector<double> const refGroup = spODFOp->getGroupBoundaries(););
  Check(soft_equiv(frequencyGroupBoundaries.begin(),
                   frequencyGroupBoundaries.end(), refGroup.begin(),
                   refGroup.end(), 1.0e-6));

  // do a check of the actual band boundary values when DBC check is on (this is
  // more expensive so we retain the option of turning it off)
  Remember(std::vector<double> const refBand = spODFOp->getBandBoundaries(););
  Check(soft_equiv(opacityCdfBandBoundaries.begin(),
                   opacityCdfBandBoundaries.end(), refBand.begin(),
                   refBand.end(), 1.0e-6));

  // assign the smart pointer
  odfmgOpacities[model][reaction] = spODFOp;

  Ensure(odfmgOpacities[model][reaction]);
}

//---------------------------------------------------------------------------//

void CDI::setEoS(const SP_EoS &in_spEoS) {
  Require(in_spEoS);
  Insist(!spEoS, "Tried to overwrite a set EoS object.!");
  // set the smart pointer
  spEoS = in_spEoS;
  Ensure(spEoS);
}

//---------------------------------------------------------------------------//

void CDI::setEICoupling(const SP_EICoupling &in_spEICoupling) {
  Require(in_spEICoupling);
  Insist(!spEICoupling, "Tried to overwrite a set EICoupling object.!");
  // set the smart pointer
  spEICoupling = in_spEICoupling;
  Ensure(spEICoupling);
}

//---------------------------------------------------------------------------//
// GET FUNCTIONS
//---------------------------------------------------------------------------//

/*!
 * \brief This fuction returns a GrayOpacity object.
 *
 * This provides the CDI with the full functionality of the interface defined in
 * GrayOpacity.hh.  For example, the host code could make the following call:
 *
 * \code
 * double newOp = spCDI1->gray()->getOpacity( 55.3, 27.4 );
 * \endcode
 *
 * The appropriate gray opacity is returned for the given model and reaction
 * type.
 *
 * \param m rtt_cdi::Model specifying the desired physics model
 * \param r rtt_cdi::Reaction specifying the desired reaction type
 */
CDI::SP_GrayOpacity CDI::gray(rtt_cdi::Model m, rtt_cdi::Reaction r) const {
  Insist(grayOpacities[m][r], "Undefined GrayOpacity!");
  return grayOpacities[m][r];
}

//----------------------------------------------------------------------------//
/*!
 * \brief This fuction returns the MultigroupOpacity object.
 *
 * This provides the CDI with the full functionality of the interface defined in
 * MultigroupOpacity.hh.  For example, the host code could make the following
 * call:
 *
 * \code
 * size_t numGroups = spCDI1->mg()->getNumGroupBoundaries();
 * \endcode
 *
 * The appropriate multigroup opacity is returned for the given reaction type.
 *
 * \param m rtt_cdi::Model specifying the desired physics model
 * \param r rtt_cdi::Reaction specifying the desired reaction type.
 */
CDI::SP_MultigroupOpacity CDI::mg(rtt_cdi::Model m, rtt_cdi::Reaction r) const {
  Insist(multigroupOpacities[m][r], "Undefined MultigroupOpacity!");
  return multigroupOpacities[m][r];
}

//----------------------------------------------------------------------------//
/*!
 * \brief This fuction returns the OdfmgOpacity object.
 *
 * This provides the CDI with the full functionality of the interface defined in
 * OdfmgOpacity.hh.  For example, the host code could make the following call:
 *
 * \code
 * size_t numGroups = spCDI1->mg()->getNumGroupBoundaries();
 * \endcode
 *
 * The appropriate multigroup opacity is returned for the given reaction type.
 *
 * \param m rtt_cdi::Model specifying the desired physics model
 * \param r rtt_cdi::Reaction specifying the desired reaction type.
 */
CDI::SP_OdfmgOpacity CDI::odfmg(rtt_cdi::Model m, rtt_cdi::Reaction r) const {
  Insist(odfmgOpacities[m][r], "Undefined OdfmgOpacity!");
  return odfmgOpacities[m][r];
}

//---------------------------------------------------------------------------//
/*!
 * \brief This fuction returns the EoS object.
 *
 * This provides the CDI with the full functionality of the interface defined in
 * EoS.hh.  For example, the host code could make the following call:
 *
 * \code
 * double Cve = spCDI1->eos()->getElectronHeatCapacity( * density, temperature );
 * \endcode
 */
CDI::SP_EoS CDI::eos() const {
  Insist(spEoS, "Undefined EoS!");
  return spEoS;
}

//---------------------------------------------------------------------------//
/*!
 * \brief This fuction returns the EICoupling object.
 *
 * This provides the CDI with the full functionality of the interface defined in
 * EICoupling.hh.  For example, the host code could make the following call:
 *
 * \code
 * double ei_coupling = spCDI1->ei_coupling()->getElectronIonCoupling( ... );
 * \endcode
 */
CDI::SP_EICoupling CDI::ei_coupling() const {
  Insist(spEICoupling, "Undefined EICoupling!");
  return spEICoupling;
}

//---------------------------------------------------------------------------//
// RESET THE CDI OBJECT
//---------------------------------------------------------------------------//

/*!
 * \brief Reset the CDI object.
 *
 * This function "clears" all data objects (GrayOpacity, MultigroupOpacity, EoS)
 * held by CDI.  After clearing, new objects can be set using the set functions.
 *
 * As stated in the set functions documentation, you are not allowed to
 * overwrite a data object with the same attributes as one that already has been
 * set.  The only way to "reset" these objects is to call CDI::reset().  Note
 * that CDI::reset() resets \b ALL of the objects stored by CDI (including group
 * boundaries).
 */
void CDI::reset() {
  Check(grayOpacities.size() == constants::num_Models);
  Check(multigroupOpacities.size() == constants::num_Models);
  Check(odfmgOpacities.size() == constants::num_Models);

  // reset the gray opacities
  for (size_t i = 0; i < constants::num_Models; ++i) {
    Check(grayOpacities[i].size() == constants::num_Reactions);
    Check(multigroupOpacities[i].size() == constants::num_Reactions);
    Check(odfmgOpacities[i].size() == constants::num_Reactions);

    for (size_t j = 0; j < constants::num_Reactions; j++) {
      // reassign the GrayOpacity shared_ptr to a null shared_ptr
      grayOpacities[i][j] = SP_GrayOpacity();

      // reassign the MultigroupOpacity shared_ptr to a null shared_ptr
      multigroupOpacities[i][j] = SP_MultigroupOpacity();

      // reassign the OdfmgOpacity shared_ptr to a null shared_ptr
      odfmgOpacities[i][j] = SP_OdfmgOpacity();

      // check it
      Check(!grayOpacities[i][j]);
      Check(!odfmgOpacities[i][j]);
    }
  }

  // empty the frequency group boundaries
  frequencyGroupBoundaries.clear();
  Check(frequencyGroupBoundaries.empty());

  // empty the opacity band boundaries
  opacityCdfBandBoundaries.clear();
  Check(opacityCdfBandBoundaries.empty());

  // reset the EoS shared_ptr
  spEoS = SP_EoS();
  Check(!spEoS);

  // reset the EoS shared_ptr
  spEICoupling = SP_EICoupling();
  Check(!spEICoupling);
}

//---------------------------------------------------------------------------//
// BOOLEAN QUERY FUNCTIONS
//---------------------------------------------------------------------------//

/*!
 * \brief Query to see if a gray opacity is set.
 */
bool CDI::isGrayOpacitySet(rtt_cdi::Model m, rtt_cdi::Reaction r) const {
  return static_cast<bool>(grayOpacities[m][r]);
}

/*!
 * \brief Query to see if a multigroup opacity is set.
 */
bool CDI::isMultigroupOpacitySet(rtt_cdi::Model m, rtt_cdi::Reaction r) const {
  return static_cast<bool>(multigroupOpacities[m][r]);
}

/*!
 * \brief Query to see if an odf multigroup opacity is set.
 */
bool CDI::isOdfmgOpacitySet(rtt_cdi::Model m, rtt_cdi::Reaction r) const {
  return static_cast<bool>(odfmgOpacities[m][r]);
}

/*!
 * \brief Query to see if an eos is set.
 */
bool CDI::isEoSSet() const { return static_cast<bool>(spEoS); }

/*!
 * \brief Query to see if an eos is set.
 */
bool CDI::isEICouplingSet() const { return static_cast<bool>(spEICoupling); }

} // end namespace rtt_cdi

//---------------------------------------------------------------------------//
// end of CDI.cc
//---------------------------------------------------------------------------//
