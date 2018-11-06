//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   ds++/scalar_unit_test.hh
 * \author Kent Grimmett Budge
 * \brief  Define uniform test function for unit testing
 * \note   Copyright (C) 2018 TRIAD, LLC. All rights reserved. */
//----------------------------------------------------------------------------//

#ifndef dsxx_scalar_unit_test_hh
#define dsxx_scalar_unit_test_hh

namespace rtt_dsxx {

//----------------------------------------------------------------------------//
/*!
 * \brief Run a scalar unit test.
 *
 * \param[in] argc Number of command line arguments
 * \param[in] argv Command line arguments
 * \param[in] release Release string
 * \param[in] lambda Lambda function defining the test.
 * \return EXIT_SUCCESS or EXIT_FAILURE as appropriate.
 */

template<typename Lambda, typename Release>
int do_scalar_unit_test(int argc, char *argv[], Release release,
                 Lambda const &lambda);

} // end namespace rtt_dsxx

#include "scalar_unit_test.i.hh"

#endif // dsxx_test_hh

//----------------------------------------------------------------------------//
// end of ds++/scalar_unit_test.hh
//----------------------------------------------------------------------------//
