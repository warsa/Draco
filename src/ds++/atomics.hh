//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   ds++/atomics.hh
 * \author Tim Kelley
 * \date   Thursday, Sept. 6, 2018, 10:50 am
 * \brief  Header file for atomic functions (until C++20)
 * \note   Copyright (C) 2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef ATOMICS_HH
#define ATOMICS_HH

#include <atomic>

namespace rtt_dsxx {

/**\brief atomically add arg to a.
 * \tparam FpT: a floating point type (integer types in std lib since C++11).
 * \param a: an atomic of type FpT that will be updated.
 * \param arg: quantity to add to a
 * \return value of a after update
 * \remark: Uses memory_order_relaxed, meaning (I think) that other atomic
 * operations on a can be moved before or after this one.
 */
template <class FpT>
FpT fetch_add(std::atomic<FpT> &a, FpT arg){
  FpT expected = a.load();
  FpT to_store = expected + arg;
  while(!a.compare_exchange_weak(expected,to_store,
      std::memory_order_relaxed)) {
    expected = a.load();
    to_store = arg + expected;
  }
  return to_store;
}

/**\brief atomically subtract a from arg.
 * \tparam FpT: a floating point type (integer types in std lib since C++11).
 * \param a: an atomic of type FpT that will be updated.
 * \param arg: quantity to subtract from a
 * \return value of a after update
 * \remark: Uses memory_order_relaxed, meaning (I think) that other atomic
 * operations on a can be moved before or after this one.
 */
template <class FpT>
FpT fetch_sub(std::atomic<FpT> &a, FpT arg){
  FpT expected = a.load();
  FpT to_store = expected - arg;
  while(!a.compare_exchange_weak(expected,to_store,
      std::memory_order_relaxed)) {
    expected = a.load();
    to_store = arg - expected;
  }
  return to_store;
}
} // namespace rtt_dsxx

#endif

// End of file
