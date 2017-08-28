//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/DracoStrings.i.hh
 * \author Kelly G. Thompson <kgt@lanl.gov
 * \date   Wednesday, Aug 23, 2017, 12:48 pm
 * \brief  Enscapulates common string manipulations (implicit template
 *         implementation).
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef rtt_dsxx_DracoStrings_i_hh
#define rtt_dsxx_DracoStrings_i_hh

#include "Assert.hh"
#include <algorithm>

namespace rtt_dsxx {

//----------------------------------------------------------------------------//
//! Convert a string into a vector of floating-point or an integral values.
template <typename T>
std::vector<T> string_to_numvec(std::string const &str,
                                std::string const &range_symbols,
                                std::string const &delimiters) {

  // for vector data, first and last char might be some delimiter character.
  // For example "[1,2,3)" or "{3.3, 4.4}"
  Insist(
      range_symbols.length() == 0 ||
          (range_symbols.length() == 2 && str[0] == range_symbols[0] &&
           str[str.length() - 1] == range_symbols[range_symbols.length() - 1]),
      "String data is malformed. It is missing { or }.");

  // remove beginning and ending braces and any extra whitespace.
  std::string const tstr = trim(str, range_symbols + " \t");

  // Convert the string into a vector<string>
  std::vector<std::string> vsdata = tokenize(tstr, delimiters);

  // convert from vector<string> to vector<T>
  std::vector<T> retval(vsdata.size());
  std::transform(vsdata.begin(), vsdata.end(), retval.begin(),
                 [](const std::string &val) { return parse_number<T>(val); });

  return retval;
}

} // namespace rtt_dsxx

#endif // namespace rtt_dsxx_DracoStrings_i_hh

//---------------------------------------------------------------------------//
// end of DracoStrings.i.hh
//---------------------------------------------------------------------------//
