//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/opstream.cc
 * \author Mike Buksas
 * \date   Thu May  1 14:42:10 2008
 * \brief
 * \note   Copyright (C) 2016-2018 Los Alamos National Security, LLC.
 *         All rights reserved.
 */
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

#include <iostream>

#include "C4_Functions.hh"
#include "opstream.hh"

namespace rtt_c4 {
using namespace std;

//---------------------------------------------------------------------------//
void opstream::mpibuf::send() {
  unsigned const pid = rtt_c4::node();
  if (pid == 0) {
    buffer_.push_back('\0');
    cout << &buffer_[0];
    buffer_.clear();

    unsigned const pids = rtt_c4::nodes();
    for (unsigned i = 1; i < pids; ++i) {
      unsigned N;
      receive(&N, 1, i);
      buffer_.resize(N);
      rtt_c4::receive(&buffer_[0], N, i);
      buffer_.push_back('\0');
      cout << &buffer_[0];
    }
  } else {
    unsigned N = buffer_.size();
    rtt_c4::send(&N, 1, 0);
    rtt_c4::send(&buffer_[0], N, 0);
  }
  buffer_.clear();
  rtt_c4::global_barrier();
}

//---------------------------------------------------------------------------//
/*virtual*/ opstream::mpibuf::int_type opstream::mpibuf::overflow(int_type c) {
  buffer_.push_back(c);
  return c;
}

//---------------------------------------------------------------------------//
void opstream::mpibuf::shrink_to_fit() { buffer_.shrink_to_fit(); }

} // end namespace rtt_c4

//---------------------------------------------------------------------------//
// end of opstream.cc
//---------------------------------------------------------------------------//
