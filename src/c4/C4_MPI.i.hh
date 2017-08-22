//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   c4/C4_MPI.i.hh
 * \author Alex R Long
 * \date   Mon Aug 21 07:47:01 2017
 * \brief  C4 MPI standard implementations.
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef c4_C4_MPI_i_hh
#define c4_C4_MPI_i_hh

#ifdef C4_MPI

namespace rtt_c4 {

//---------------------------------------------------------------------------//
template <typename T>
DLL_PUBLIC_c4 void send_is_custom(C4_Req &request, const T *buffer, int size,
                                  int destination, int tag,
                                  MPI_Datatype custom_type) {
  Require(!request.inuse());

  // set the request
  request.set();

  Remember(int const retval =)
      MPI_Issend(const_cast<T *>(buffer), size, T::MPI_Type, destination, tag,
                 communicator, &request.r());
  Check(retval == MPI_SUCCESS);

  return;
}

//---------------------------------------------------------------------------//

template <typename T>
DLL_PUBLIC_c4 void receive_async_custom(C4_Req &request, T *buffer, int size,
                                        int source, int tag,
                                        MPI_Datatype custom_type) {
  Require(!request.inuse());
  Remember(int custom_mpi_type_size);
  Remember(MPI_Type_size(T::MPI_Type, &custom_mpi_type_size));
  Require(custom_mpi_type_size == sizeof(T));

  // set the request
  request.set();

  // post an MPI_Irecv
  Remember(int const retval =) MPI_Irecv(buffer, size, T::MPI_Type, source, tag,
                                         communicator, &request.r());
  Check(retval == MPI_SUCCESS);
  return;
}

} // end namespace rtt_c4

#endif // C4_MPI

#endif // c4_C4_MPI_i_hh

//---------------------------------------------------------------------------//
// end of c4/C4_MPI.i.hh
//---------------------------------------------------------------------------//
