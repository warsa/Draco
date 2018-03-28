//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/runtime_check.cc
 * \author Kent Grimmett Budge
 * \date   Wed Mar 28 07:58:48 2018
 * \brief  Member definitions of class runtime_check
 * \note   Copyright (C) 2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include <exception>

#include "C4_Functions.hh"
#include "ds++/DracoStrings.hh"

namespace rtt_dsxx {

//-------------------------------------------------------------------------//
void runtime_check(bool condition, char const *message) noexcept(false) {
  Require(message != nullptr);

  unsigned sum = !condition;
  rtt_c4::global_sum(sum);
  if (sum != 0) // some processors failed the condition
  {
    throw std::runtime_error("runtime error detected on " +
                             rtt_dsxx::to_string(sum) + " processor(s): " +
                             message);
  }
}

} // end namespace rtt_dsxx

//---------------------------------------------------------------------------//
// end of ds++/runtime_check.cc
//---------------------------------------------------------------------------//
