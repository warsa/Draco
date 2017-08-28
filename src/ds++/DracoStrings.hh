//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/DracoStrings.hh
 * \author Kelly G. Thompson <kgt@lanl.gov
 * \date   Wednesday, Aug 23, 2017, 12:48 pm
 * \brief  Enscapulates common string manipulations.
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef rtt_dsxx_DracoStrings_hh
#define rtt_dsxx_DracoStrings_hh

#include "ds++/config.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace rtt_dsxx {

//----------------------------------------------------------------------------//
/*!
 * \brief Convert a number into a string
 *
 * \param[in] num The integral or float-point type number that will be converted
 *                to a string.
 * \param[in] precision How many digits will be preserved?
 * \return A string representation of the numeric value
 *
 * \sa http://public.research.att.com/~bs/bs_faq2.html
 */
template <typename T>
std::string to_string(T const num, unsigned int const precision = 23) {
  std::stringstream s;
  s.precision(precision);
  s << num;
  return s.str();
}

//----------------------------------------------------------------------------//
/*!
 * \brief trim whitespace (or other characters) from before and after main text.
 *
 * \param[in] str The string that will be processed
 * \param[in] whitespace A set of characters that will be removed.
 * \return A new, probably shortened, string without unwanted leading/training
 *         characters.
 */
DLL_PUBLIC_dsxx std::string trim(std::string const &str,
                                 std::string const &whitespace = " \t");

//----------------------------------------------------------------------------//
/*!
 * \brief Removes all specified characters from a string.
 *
 * \param[in] str The string that will be processed
 * \param[in] chars_to_remove A set of characters (as a std::string) that will
 *         be removed.
 * \return A new, possibly shortened, string that does not contain the unwanted
 *         characters.
 */
DLL_PUBLIC_dsxx std::string prune(std::string const &orig_str,
                                  std::string const &chars_to_remove);

//----------------------------------------------------------------------------//
/*!
 * \brief Split a string into a vector<string> using a specified delimiter.
 *
 * \param[in] str The string that will be split
 * \param[in] delimiter Set of character that separate each 'word'.
 * \param[in] keepEmptyStrings Should empty values be saved in the result?
 * \return a vector of strings. The delimiter is always removed.
 */
DLL_PUBLIC_dsxx std::vector<std::string>
tokenize(std::string const &str, std::string const &delimiters = " ",
         bool keepEmptyStrings = false);

//----------------------------------------------------------------------------//
/*!
 * \brief Convert a string into a floating-point type or an integral type.
 *
 * \param[in] str The string that contains a number.
 * \return A numeric value.
 *
 * The template type will be deduced based on the return type ("-> T").
 *
 * To avoid invalid conversions, consider:
 *
 * \code
 * try {
 *    x = stoi(y);
 * } catch(std::invalid_argument& e) {
 *    // if no conversion could be performed
 * } catch(std::out_of_range& e) {
 *    // if the converted value would fall out of the range of the result type
 *    // or if the underlying function (std::strtol or std::strtoull) sets
 *    // errno to ERANGE.
 * } catch(...) {
 *   // everything else
 * }
 * \endcode
 */
template <typename T> auto parse_number(std::string const &str) -> T;

// specializations for these types are devined in DracoStrings.cc
template <> auto parse_number<int>(std::string const &str) -> int;
template <> auto parse_number<long>(std::string const &str) -> long;
template <>
auto parse_number<unsigned long>(std::string const &str) -> unsigned long;
template <> auto parse_number<float>(std::string const &str) -> float;
template <> auto parse_number<double>(std::string const &str) -> double;

//----------------------------------------------------------------------------//
/*!
 * \brief Convert a string into a vector of floating-point or an integral
 *        values.
 *
 * \param[in] str The string that contains a number.
 * \param[in] range_symbols Parenthesis or brances that mark the begining or end
 *         of the value range. Example "()" or "{}".
 * \param[in] delimiter A character that separates each numeric entry.
 * \return A vector of numeric values.
 *
 * std::string foo = "{ 1.0, 2.0, 3.0 }" will be converted to
 * std::vector<double> bar = { 1.0, 2.0, 3.0 }
 */
template <typename T>
DLL_PUBLIC_dsxx std::vector<T>
string_to_numvec(std::string const &str,
                 std::string const &range_symbols = "{}",
                 std::string const &delimiters = ",");

//----------------------------------------------------------------------------//
/*!
 * \brief Parse an ostream and create a map containing [word]:[num_occurances]
 *
 * \param[in] data An ostringstream that will be parsed
 * \param[in] verbose
 * \return a map<string,uint> that contains [word]:[num_occurances]
 */
DLL_PUBLIC_dsxx std::map<std::string, unsigned>
get_word_count(std::ostringstream const &data, bool verbose = false);

//----------------------------------------------------------------------------//
/*!
 * \brief Parse a file and create a map containing [word]:[num_occurances]
 *
 * \param[in] filename The name of the file (full path recommended) that will be
 *               parsed.
 * \param[in] verbose
 * \return a map<string,uint> that contains [word]:[num_occurances]
 */
DLL_PUBLIC_dsxx std::map<std::string, unsigned>
get_word_count(std::string const &filename, bool verbose = false);

} // namespace rtt_dsxx

// Template functions
#include "DracoStrings.i.hh"

#endif // rtt_dsxx_DracoStrings_hh

//---------------------------------------------------------------------------//
// end of DracoStrings.hh
//---------------------------------------------------------------------------//
