//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   cdi_analytic/test/tstAnalytic_Gray_Opacity.cc
 * \author Thomas M. Evans
 * \date   Mon Sep 24 12:08:55 2001
 * \brief  Analytic_Gray_Opacity test.
 * \note   Copyright (C) 2016-2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "cdi_analytic_test.hh"
#include "cdi/CDI.hh"
#include "cdi_analytic/Analytic_Gray_Opacity.hh"
#include "cdi_analytic/Compound_Analytic_MultigroupOpacity.hh"
#include "ds++/Release.hh"
#include "ds++/ScalarUnitTest.hh"
#include <sstream>

using namespace std;

using rtt_cdi::CDI;
using rtt_cdi::GrayOpacity;
using rtt_cdi_analytic::Analytic_Gray_Opacity;
using rtt_cdi_analytic::Analytic_Opacity_Model;
using rtt_cdi_analytic::Compound_Analytic_MultigroupOpacity;
using rtt_cdi_analytic::Constant_Analytic_Opacity_Model;
using rtt_cdi_analytic::Polynomial_Analytic_Opacity_Model;
using rtt_dsxx::soft_equiv;
using std::dynamic_pointer_cast;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

void constant_test(rtt_dsxx::UnitTest &ut) {
  // make an analytic gray opacity that returns the total opacity for a constant
  // model
  const double constant_opacity = 5.0;

  shared_ptr<Analytic_Opacity_Model> model(
      new Constant_Analytic_Opacity_Model(constant_opacity));

  Analytic_Gray_Opacity anal_opacity(model, rtt_cdi::TOTAL);

  // link to GrayOpacity
  GrayOpacity *grayp = &anal_opacity;

  // Some basic tests
  if (grayp->data_in_tabular_form())
    ITFAILS;
  if (grayp->getEnergyPolicyDescriptor() != "gray")
    ITFAILS;
  if (grayp->getDataDescriptor() != "Analytic Gray Total")
    ITFAILS;
  if (grayp->getReactionType() != rtt_cdi::TOTAL)
    ITFAILS;
  if (grayp->getModelType() != rtt_cdi::ANALYTIC)
    ITFAILS;
  if (grayp->getOpacityModelType() != rtt_cdi::ANALYTIC_TYPE)
    ITFAILS;
  if (typeid(grayp) != typeid(GrayOpacity *))
    ITFAILS;
  if (typeid(*grayp) != typeid(Analytic_Gray_Opacity))
    ITFAILS;

  {
    Analytic_Gray_Opacity analyt_opacity(model, rtt_cdi::ABSORPTION);
    if (analyt_opacity.getDataDescriptor() != "Analytic Gray Absorption")
      ITFAILS;
  }
  {
    Analytic_Gray_Opacity analyt_opacity(model, rtt_cdi::TOTAL);
    if (analyt_opacity.getDataDescriptor() != "Analytic Gray Total")
      ITFAILS;
  }

  // check the output
  vector<double> T(10);
  vector<double> rho(10);

  for (size_t i = 0; i < T.size(); i++) {
    T[i] = 0.1 + i / 100.0;
    rho[i] = 1.0 + i / 10.0;

    if (!rtt_dsxx::soft_equiv(grayp->getOpacity(T[i], rho[i]),
                              constant_opacity))
      ITFAILS;
  }

  vector<double> opacity_T = grayp->getOpacity(T, 3.0);
  vector<double> opacity_rho = grayp->getOpacity(1.0, rho);
  vector<double> ref(10, constant_opacity);

  if (opacity_T != ref)
    ITFAILS;
  if (opacity_rho != ref)
    ITFAILS;

  return;
}

//---------------------------------------------------------------------------//

void user_defined_test(rtt_dsxx::UnitTest &ut) {
  // make the user defined Marshak model
  shared_ptr<Analytic_Opacity_Model> model(
      new rtt_cdi_analytic_test::Marshak_Model(10.0));

  Analytic_Gray_Opacity anal_opacity(model, rtt_cdi::TOTAL);
  GrayOpacity *grayp = &anal_opacity;

  vector<double> T(6);
  vector<double> rho(6);
  {
    T[0] = .993;
    T[1] = .882;
    T[2] = .590;
    T[3] = .112;
    T[4] = .051;
    T[5] = .001;

    std::fill(rho.begin(), rho.end(), 3.0);
  }

  vector<double> opacities = grayp->getOpacity(T, rho[0]);
  if (opacities.size() != 6)
    ITFAILS;

  for (size_t i = 0; i < T.size(); i++) {
    double ref = 10.0 / (T[i] * T[i] * T[i]);
    double error = fabs(grayp->getOpacity(T[i], rho[i]) - ref);
    double error_field = fabs(opacities[i] - ref);

    if (error > 1.0e-12 * ref)
      ITFAILS;
    if (error_field > 1.0e-12 * ref)
      ITFAILS;
  }

  // check to make sure we can't unpack an unregistered analytic model
  vector<char> packed = anal_opacity.pack();

  bool caught = false;
  try {
    Analytic_Gray_Opacity ngray(packed);
  } catch (const rtt_dsxx::assertion &err) {
    caught = true;
    ostringstream message;
    message << "Caught the following assertion, " << err.what();
    PASSMSG(message.str());
  }
  if (!caught)
    FAILMSG("Failed to catch unregistered analyic model assertion");
  return;
}

//---------------------------------------------------------------------------//

void CDI_test(rtt_dsxx::UnitTest &ut) {
  // lets make a marshak model gray opacity for scattering and absorption
  shared_ptr<const GrayOpacity> absorption;
  shared_ptr<const GrayOpacity> scattering;

  // lets make two models
  shared_ptr<Analytic_Opacity_Model> amodel(
      new Polynomial_Analytic_Opacity_Model(0.0, 100.0, -3.0, 0.0));
  shared_ptr<Analytic_Opacity_Model> smodel(
      new Constant_Analytic_Opacity_Model(1.0));

  if (!soft_equiv(amodel->calculate_opacity(2.0, 3.0, 4.0),
                  100.0 / (2.0 * 2.0 * 2.0)))
    FAILMSG("FAILED to calculate grey absorption opacity");
  if (!soft_equiv(smodel->calculate_opacity(1.0, 1.0, 1.0), 1.0))
    FAILMSG("FAILED to calculate grey scattering opacity");

  absorption.reset(
      new const Analytic_Gray_Opacity(amodel, rtt_cdi::ABSORPTION));
  scattering.reset(
      new const Analytic_Gray_Opacity(smodel, rtt_cdi::SCATTERING));
  if (absorption->getDataDescriptor() != "Analytic Gray Absorption")
    ITFAILS;
  if (scattering->getDataDescriptor() != "Analytic Gray Scattering")
    ITFAILS;
  {
    shared_ptr<const GrayOpacity> total(
        new const Analytic_Gray_Opacity(smodel, rtt_cdi::TOTAL));

    if (total->getDataDescriptor() != "Analytic Gray Total")
      ITFAILS;
  }

  if (!absorption)
    FAILMSG("Failed to build absorption analytic opacity");
  if (!scattering)
    FAILMSG("Failed to build scattering analytic opacity");

  // make a CDI for scattering and absorption
  CDI cdi;
  cdi.setGrayOpacity(scattering);
  cdi.setGrayOpacity(absorption);

  // now check some data
  vector<double> T(6);
  vector<double> rho(6, 3.0);
  {
    T[0] = .993;
    T[1] = .882;
    T[2] = .590;
    T[3] = .112;
    T[4] = .051;
    T[5] = .001;

    std::fill(rho.begin(), rho.end(), 3.0);
  }

  for (size_t i = 0; i < T.size(); i++) {
    double ref = 100.0 / (T[i] * T[i] * T[i]);
    rtt_cdi::Model model = rtt_cdi::ANALYTIC;
    rtt_cdi::Reaction abs = rtt_cdi::ABSORPTION;
    rtt_cdi::Reaction scat = rtt_cdi::SCATTERING;

    double error = fabs(cdi.gray(model, abs)->getOpacity(T[i], rho[i]) - ref);

    if (error > 1.0e-12 * ref)
      ITFAILS;

    error = fabs(cdi.gray(model, scat)->getOpacity(T[i], rho[i]) - 1.0);

    if (error > 1.0e-12)
      ITFAILS;
  }

  // Test the get_parameters() member function
  {
    std::vector<double> params(amodel->get_parameters());

    std::vector<double> expectedValue(11);
    expectedValue[0] = 0.0;
    expectedValue[1] = 100.0;
    expectedValue[2] = -3.0;
    expectedValue[3] = 0.0;
    expectedValue[4] = 0.0;
    expectedValue[5] = 1.0;
    expectedValue[6] = 1.0;
    expectedValue[7] = 1.0;
    expectedValue[8] = 0.0;
    expectedValue[9] = 0.0;
    expectedValue[10] = 0.0;

    double const tol(1.0e-12);

    if (params.size() != expectedValue.size())
      ITFAILS;

    if (soft_equiv(params.begin(), params.end(), expectedValue.begin(),
                   expectedValue.end(), tol))
      PASSMSG(
          "get_parameters() returned the analytic expression coefficients.");
    else
      FAILMSG("get_parameters() did not return the analytic expression "
              "coefficients.");
  }
  return;
}

//---------------------------------------------------------------------------//

void packing_test(rtt_dsxx::UnitTest &ut) {
  // test the packing
  vector<char> packed;
  {
    // lets make two models
    shared_ptr<Analytic_Opacity_Model> amodel(
        new Polynomial_Analytic_Opacity_Model(0.0, 100.0, -3.0, 0.0));

    Analytic_Gray_Opacity absorption(amodel, rtt_cdi::ABSORPTION);

    packed = absorption.pack();
  }

  // now unpack and test
  Analytic_Gray_Opacity ngray(packed);

  // now check some data
  vector<double> T(6);
  vector<double> rho(6, 3.0);
  {
    T[0] = .993;
    T[1] = .882;
    T[2] = .590;
    T[3] = .112;
    T[4] = .051;
    T[5] = .001;

    std::fill(rho.begin(), rho.end(), 3.0);
  }

  for (size_t i = 0; i < T.size(); i++) {
    double ref = 100.0 / (T[i] * T[i] * T[i]);

    double error = fabs(ngray.getOpacity(T[i], rho[i]) - ref);

    if (error > 1.0e-12 * ref)
      ITFAILS;
  }

  if (ngray.getReactionType() != rtt_cdi::ABSORPTION)
    ITFAILS;
  if (ngray.getModelType() != rtt_cdi::ANALYTIC)
    ITFAILS;

  if (ut.numFails == 0)
    PASSMSG("Analytic_Gray_Opacity packing test passes.");

  return;
}

//---------------------------------------------------------------------------//

void type_test(rtt_dsxx::UnitTest &ut) {
  // make an analytic gray opacity that returns the total opacity for a constant
  // model
  const double constant_opacity = 5.0;

  shared_ptr<Analytic_Opacity_Model> model(
      new Constant_Analytic_Opacity_Model(constant_opacity));

  shared_ptr<GrayOpacity> op(new Analytic_Gray_Opacity(model, rtt_cdi::TOTAL));
  shared_ptr<Analytic_Gray_Opacity> opac;

  if (typeid(*op) == typeid(rtt_cdi_analytic::Analytic_Gray_Opacity)) {
    PASSMSG("RTTI type info is correct for shared_ptr to GrayOpacity.");
    opac = dynamic_pointer_cast<Analytic_Gray_Opacity>(op);
  }

  vector<double> parm = opac->get_Analytic_Model()->get_parameters();

  if (parm.size() != 1)
    ITFAILS;
  if (!soft_equiv(constant_opacity, parm.front()))
    ITFAILS;

  // another way to do this
  Compound_Analytic_MultigroupOpacity *m =
      dynamic_cast<Compound_Analytic_MultigroupOpacity *>(&*op);
  Analytic_Gray_Opacity *o = dynamic_cast<Analytic_Gray_Opacity *>(&*op);

  if (m)
    ITFAILS;
  if (!o)
    ITFAILS;
  if (typeid(*o) != typeid(rtt_cdi_analytic::Analytic_Gray_Opacity))
    ITFAILS;
}

//---------------------------------------------------------------------------//
void default_behavior_tests(rtt_dsxx::UnitTest &ut) {
  // make an analytic gray opacity that returns the total opacity for a constant
  // model
  const double constant_opacity = 5.0;

  shared_ptr<Analytic_Opacity_Model> model(
      new Constant_Analytic_Opacity_Model(constant_opacity));

  Analytic_Gray_Opacity opac(model, rtt_cdi::TOTAL);

  // There is no data file associated with this analytic opacity model.
  // This function should return an empty string.
  {
    std::string datafilename = opac.getDataFilename();
    std::string expectedValue;

    if (datafilename.length() == 0 && expectedValue.length() == 0)
      PASSMSG("getDataFilename() returned an empty string.");
    else
      FAILMSG("getDataFilename() did not return an empty string.");
  }

  // There is no density grid associated with this analytic opacity model.
  // This function should return an empty vector<double>.
  {
    vector<double> densityGrid = opac.getDensityGrid();
    vector<double> expectedValue;

    if (densityGrid == expectedValue)
      PASSMSG("getDensityGrid() returned an empty string.");
    else
      FAILMSG("getDensityGrid() did not return an empty string.");
  }

  // There is no density grid associated with this analytic opacity model.
  // This function should return an empty vector<double>.
  {
    vector<double> densityGrid = opac.getDensityGrid();
    vector<double> expectedValue;

    if (opac.getNumDensities() == 0)
      PASSMSG("getNumDensities() returned 0.");
    else
      FAILMSG("getNumDensities() did not return 0.");

    if (densityGrid == expectedValue)
      PASSMSG("getDensityGrid() returned an empty vector.");
    else
      FAILMSG("getDensityGrid() did not return an empty vector.");
  }

  // There is no temperature grid associated with this analytic opacity model.
  // This function should return an empty vector<double>.
  {
    vector<double> temperatureGrid = opac.getTemperatureGrid();
    vector<double> expectedValue;

    if (opac.getNumTemperatures() == 0)
      PASSMSG("getNumTemperatures() returned 0.");
    else
      FAILMSG("getNumTemperatures() did not return 0.");

    if (temperatureGrid == expectedValue)
      PASSMSG("getTemperatureGrid() returned an empty vector.");
    else
      FAILMSG("getTemperatureGrid() did not return an empty vector.");
  }

  return;
}

//---------------------------------------------------------------------------//

int main(int argc, char *argv[]) {
  rtt_dsxx::ScalarUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    constant_test(ut);
    user_defined_test(ut);
    CDI_test(ut);
    packing_test(ut);
    type_test(ut);
    default_behavior_tests(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
// end of tstAnalytic_Gray_Opacity.cc
//---------------------------------------------------------------------------//
