//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/RTT_Draco_Mesh_Reader.hh
 * \author Ryan Wollaeger <wollaeger@lanl.gov>
 * \date   Friday, Jul 13, 2018, 08:38 am
 * \brief  RTT_Draco_Mesh_Reader header file.
 * \note   Copyright (C) 2018-2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef rtt_mesh_RTT_Draco_Mesh_Reader_hh
#define rtt_mesh_RTT_Draco_Mesh_Reader_hh

#include "Draco_Mesh_Reader.hh"
#include "RTT_Format_Reader/RTT_Format_Reader.hh"

namespace rtt_mesh {

//===========================================================================//
/*!
 * \class RTT_Draco_Mesh_Reader
 *
 * \brief Wrap RTT file format reader and provide data to Draco_Mesh_Builder.
 */
//===========================================================================//

class RTT_Draco_Mesh_Reader : public Draco_Mesh_Reader {
private:
  // >>> DATA

  const std::string filename;

  std::shared_ptr<rtt_RTT_Format_Reader::RTT_Format_Reader> rtt_reader;

public:
  //! Constructor
  explicit RTT_Draco_Mesh_Reader(const std::string filename_);

  // >>> SERVICES

  void read_mesh();

  // >>> ACCESSORS

  bool get_use_face_types() const { return false; }
  unsigned get_numdim() const { return rtt_reader->get_dims_ndim(); }
  size_t get_numcells() const { return rtt_reader->get_dims_ncells(); }
  size_t get_numnodes() const { return rtt_reader->get_dims_nnodes(); }
  std::vector<double> get_nodecoord(size_t node) const {
    return rtt_reader->get_nodes_coords(node);
  }
  std::vector<unsigned> get_cellnodes(size_t cell) const {
    return rtt_reader->get_cells_nodes(cell);
  }
  [[noreturn]] std::vector<unsigned> get_cellfacenodes(size_t /*cell*/,
                                                       size_t /*face*/) const {
    Insist(false, "cell-face nodes not implemented with RTT reader.");
  } size_t get_numsides() const {
    return rtt_reader->get_dims_nsides();
  }
  unsigned get_sideflag(size_t side) const {
    return rtt_reader->get_sides_flags(side, 0);
  }
  std::vector<unsigned> get_sidenodes(size_t side) const {
    return rtt_reader->get_sides_nodes(side);
  }

  // accessors with deferred implementation
  DLL_PUBLIC_mesh unsigned get_celltype(size_t cell) const;
  DLL_PUBLIC_mesh size_t get_sidetype(size_t side) const;
};

} // end namespace rtt_mesh

#endif // rtt_mesh_RTT_Draco_Mesh_Reader_hh

//---------------------------------------------------------------------------//
// end of mesh/RTT_Draco_Mesh_Reader.hh
//---------------------------------------------------------------------------//
