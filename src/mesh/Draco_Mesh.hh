//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/Draco_Mesh.hh
 * \date   April 2018
 * \brief  Draco_Mesh class header file.
 * \note   Copyright (C) 2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef rtt_mesh_Draco_Mesh_hh
#define rtt_mesh_Draco_Mesh_hh

#include "mesh_element/Geometry.hh"
#include <map>
#include <vector>

namespace rtt_mesh {

//===========================================================================//
/*!
 * \class Draco_Mesh
 *
 * \brief General unstructured mesh class.
 *
 * The Draco_Mesh class takes cell-node (or cell-vertex) data, and generates a
 * mesh with layout (cell adjacency) information.  This class also provides
 * basic services, including access to cell information.  This mesh is based on
 * an unstructured mesh implementation by Kent Budge.
 *
 * The cell_type_ vector (argument to the constructor) is currently taken to
 * be the number of nodes per cell.
 */
//===========================================================================//

class Draco_Mesh {
public:
  // >>> TYPEDEFS
  typedef std::map<std::pair<unsigned, unsigned>,
                   std::vector<std::vector<double>>>
      Cell_Face_Pair_Coord_Map;
  typedef rtt_mesh_element::Geometry Geometry;

private:
  // >>> DATA

  // Dimension
  unsigned dimension;

  // Geometry enumeration
  Geometry geometry;

  // Number of cells
  unsigned num_cells;

  // Number of nodes
  unsigned num_nodes;

  // Layout of mesh: vector index is cell index, vector element is
  // description of cell's adjacency to other cells in the mesh.
  //  Layout cell_to_cell_linkage;
  // TODO: update this to a full layout class (commented above).
  std::map<unsigned, std::vector<unsigned>> node_to_cell_map;

  // Map of cell+face index pair to vector of node coordinates for the face
  Cell_Face_Pair_Coord_Map cell_face_to_node_coord_map;

public:
  //! Constructor.
  Draco_Mesh(unsigned dimension_, Geometry geometry_,
             const std::vector<unsigned> &cell_type_,
             const std::vector<unsigned> &cell_to_node_linkage_,
             const std::vector<unsigned> &side_set_flag_,
             const std::vector<unsigned> &side_node_count_,
             const std::vector<unsigned> &side_to_node_linkage_,
             const std::vector<double> &coordinates_,
             const std::vector<unsigned> &global_node_number_);

  // >>> ACCESSORS

  unsigned get_dimension() const { return dimension; }
  Geometry get_geometry() const { return geometry; }
  unsigned get_num_cells() const { return num_cells; }
  //  const Layout &get_cc_linkage() const { return cell_to_cell_linkage; }

  // >>> SERVICES

private:
  // >>> SUPPORT FUNCTIONS

  //! Calculate the cell-to-cell linkage (layout)
  // TODO: add layout class and complete temporary version of this function.
  void compute_cell_to_cell_linkage(
      const std::vector<unsigned> &cell_type,
      const std::vector<unsigned> &cell_to_node_linkage);

  //! Calculate the mapping from cell-face pairs to node coordinates
  void
  compute_cell_face_to_node_coord_map(const std::vector<double> &coordinates);
};

} // end namespace rtt_mesh

#endif // rtt_mesh_Draco_Mesh_hh

//---------------------------------------------------------------------------//
// end of mesh/Draco_Mesh.hh
//---------------------------------------------------------------------------//
