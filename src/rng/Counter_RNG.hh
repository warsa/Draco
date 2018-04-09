//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   rng/Counter_RNG.hh
 * \author Peter Ahrens
 * \date   Fri Aug 3 16:53:23 2012
 * \brief  Declaration of class Counter_RNG.
 * \note   Copyright (C) 2016-2018 Los Alamos National Security, LLC.
 *         All rights reserved */
//---------------------------------------------------------------------------//

#ifndef Counter_RNG_hh
#define Counter_RNG_hh

#include "rng/config.h"

#ifdef _MSC_FULL_VER
// Engines have multiple copy constructors, quite legal C++, disable MSVC
// complaint.
#pragma warning(disable : 4521)
#endif

#if defined(__ICC)
// Suppress Intel's "unrecognized preprocessor directive" warning, triggered by
// use of #warning in Random123/features/sse.h.
#pragma warning disable 11
#endif

#if defined(__GNUC__) && !defined(__clang__)

/*
#if (RNG_GNUC_VERSION >= 40204) && !defined(__ICC) && !defined(NVCC)
// Suppress GCC's "unused parameter" warning, about lhs and rhs in sse.h, and an
// "unused local typedef" warning, from a pre-C++11 implementation of a static
// assertion in compilerfeatures.h.
*/
#pragma GCC diagnostic push
#if (RNG_GNUC_VERSION >= 70000)
#pragma GCC diagnostic ignored "-Wexpansion-to-defined"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexpansion-to-defined"
#endif

#include "Random123/threefry.h"
#include "uniform.hpp"

#ifdef __clang__
// Restore clang diagnostics to previous state.
#pragma clang diagnostic pop
#endif

/* #if (RNG_GNUC_VERSION >= 40600) */
#if defined(__GNUC__) && !defined(__clang__)
/* && (RNG_GNUC_VERSION >= 70000) */
// Restore GCC diagnostics to previous state.
#pragma GCC diagnostic pop
#endif

#include "ds++/Data_Table.hh"
#include <algorithm>

namespace rtt_rng {

// Forward declaration.
class Counter_RNG;

// Select a particular counter-based random number generator from Random123.
typedef r123::Threefry2x64 CBRNG;

// Counter and key types.
typedef CBRNG::ctr_type ctr_type;
typedef CBRNG::key_type key_type;

#define CBRNG_DATA_SIZE 4

namespace // anonymous
{

//---------------------------------------------------------------------------//
/*! \brief Generate a nearly-unique identifier.
 *
 * Given a pointer to RNG state data, this function generates a 64-bit
 * identifier unique to this generator but not to the specific position of its
 * RNG stream.  In other words, the identifier associated with a given generator
 * will not change as random numbers are generated from it.
 *
 * This function simply applies the chosen counter-based RNG to a shuffled
 * version of the RNG seed, stream number, and spawn indicator and then returns
 * the lower 64 bits of the result.
 */
static inline uint64_t _get_unique_num(const ctr_type::value_type *const data) {
  CBRNG hash;
  const ctr_type ctr = {{data[3], data[2]}};
  const key_type key = {{data[1] >> 32, 0}};
  const ctr_type result = hash(ctr, key);
  return result[0];
}

//---------------------------------------------------------------------------//
/*! \brief Generate a random double.
 *
 * Given a pointer to RNG state data, this function returns a random double in
 * the open interval (0, 1)---i.e., excluding the endpoints.
 */
static inline double _ran(ctr_type::value_type *const data) {
  CBRNG rng;

  // Assemble a counter from the first two elements in data.
  ctr_type ctr = {{data[0], data[1]}};

  // Assemble a key from the last two elements in data.
  const key_type key = {{data[2], data[3]}};

  // Invoke the counter-based rng.
  const ctr_type result = rng(ctr, key);

  // Increment the counter.
  ctr.incr();

  // Copy the updated counter back into data.
  data[0] = ctr[0];
  data[1] = ctr[1];

  // Convert the first 64 bits of the RNG output into a double-precision value
  // in the open interval (0, 1) and return it.
  return r123::u01fixedpt<double, ctr_type::value_type>(result[0]);
}

} // end anonymous

//===========================================================================//
/*!
 * \class Counter_RNG_Ref
 * \brief A reference to a Counter_RNG.
 *
 * Counter_RNG_Ref provides an interface to a counter-based random number
 * generator from the Random123 library from D. E. Shaw Research
 * (http://www.deshawresearch.com/resources_random123.html).  Unlike
 * Counter_RNG, Counter_RNG_Ref doesn't own its own RNG state (i.e., key and
 * counter); instead, it operates using a data block specified during
 * construction.
 */
//===========================================================================//
class Counter_RNG_Ref {
public:
  //! Constructor.  db and de specify the extents of an RNG state block.
  Counter_RNG_Ref(ctr_type::value_type *const db,
                  ctr_type::value_type *const de)
      : data(db, de) {
    Require(std::distance(db, de) * sizeof(ctr_type::value_type) ==
            sizeof(ctr_type) + sizeof(key_type));
  }

  //! Return a random double in the open interval (0, 1).
  double ran() const { return _ran(data.access()); }

  //! Return the stream number.
  uint64_t get_num() const { return data[2]; }

  //! Return a unique identifier for this generator.
  uint64_t get_unique_num() const { return _get_unique_num(data.access()); }

  //! Is this Counter_RNG_Ref a reference to rng?
  inline bool is_alias_for(Counter_RNG const &rng) const;

private:
  mutable rtt_dsxx::Data_Table<ctr_type::value_type> data;
};

//===========================================================================//
/*!
 * \class Counter_RNG
 * \brief A counter-based random-number generator.
 *
 * Counter_RNG provides an interface to a counter-based random number generator
 * from the Random123 library from D. E. Shaw Research
 * (http://www.deshawresearch.com/resources_random123.html).
 *
 * Similarly, Rnd_Control is a friend of Counter_RNG because initializing a
 * generator requires access to private data that should not be exposed through
 * the public interface.  Rnd_Control takes no responsibility for instantiating
 * Counter_RNGs itself, and since copying Counter_RNGs is disabled (via a
 * private copy constructor), an Rnd_Control must be able to initialize a
 * generator that was instantiated outside of its control.
 */
//===========================================================================//
class Counter_RNG {

  /* * Counter_RNG_Ref is a friend of Counter_RNG because spawning a new generator
   * modifies both the parent and the child generator in ways that should not be
   * exposed through the public interface of Counter_RNG.
   */
//  friend class Counter_RNG_Ref;
  friend class Rnd_Control;

public:
  typedef ctr_type::const_iterator const_iterator;

  /*! \brief Default constructor.
   *
   * This default constructor is invoked when a client wants to create a
   * Counter_RNG but delegate its initialization to an Rnd_Control object.
   */
  Counter_RNG() {
    Require(sizeof(data) == sizeof(ctr_type) + sizeof(key_type));
  }

  //! Construct a Counter_RNG using a seed and stream number.
  Counter_RNG(const uint32_t seed, const uint64_t streamnum) {
    initialize(seed, streamnum);
  }

  //! Create a new Counter_RNG from data.
  Counter_RNG(const ctr_type::value_type *const begin,
              const ctr_type::value_type *const end) {
    Require(std::distance(begin, end) * sizeof(ctr_type::value_type) ==
            sizeof(ctr_type) + sizeof(key_type));

    std::copy(begin, end, data);
  }

  //! Return a random double in the interval (0, 1).
  double ran() const { return _ran(data); }

  //! Return the stream number.
  uint64_t get_num() const { return data[2]; }

  //! Return a unique identifier for this generator.
  uint64_t get_unique_num() const { return _get_unique_num(data); }

  //! Return an iterator to the beginning of the state block.
  const_iterator begin() const { return data; }

  //! Return an iterator to the end of the state block.
  const_iterator end() const { return data + size(); }

  //! Test for equality.
  bool operator==(Counter_RNG const &rhs) const {
    return std::equal(begin(), end(), rhs.begin());
  }

  //! Test for inequality.
  bool operator!=(Counter_RNG const &rhs) const {
    return !std::equal(begin(), end(), rhs.begin());
  }

  //! Return a Counter_RNG_Ref corresponding to this Counter_RNG.
  Counter_RNG_Ref ref() const { return Counter_RNG_Ref(data, data + size()); }

  //! Return the size of this Counter_RNG.
  size_t size() const { return size_bytes() / sizeof(ctr_type::value_type); }

  //! Return the size of this Counter_RNG in bytes.
  size_t size_bytes() const { return sizeof(data); }

private:
  mutable ctr_type::value_type data[CBRNG_DATA_SIZE];

  //! Private copy constructor.
  Counter_RNG(const Counter_RNG &);

  //! Private assignment operator.
  Counter_RNG &operator=(const Counter_RNG &);

  //! Initialize internal state from a seed and stream number.
  inline void initialize(const uint32_t seed, const uint64_t streamnum);
};

//---------------------------------------------------------------------------//
// Implementation
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! Is this Counter_RNG_Ref a reference to rng?
inline bool Counter_RNG_Ref::is_alias_for(Counter_RNG const &rng) const {
  return rng.begin() == data.access();
}

//---------------------------------------------------------------------------//
//! \brief Initialize internal state from a seed and stream number.
inline void Counter_RNG::initialize(const uint32_t seed,
                                    const uint64_t streamnum) {
  // Low bits of the counter.
  data[0] = 0;

  // High bits of the counter; used for the seed.
  data[1] = static_cast<uint64_t>(seed) << 32;

  // Low bits of the key; used for the stream number.
  data[2] = streamnum;

  // High bits of the key; unused at present
  data[3] = 0;
}

} // end namespace rtt_rng

#endif

//---------------------------------------------------------------------------//
// end Counter_RNG.hh
//---------------------------------------------------------------------------//
