//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/Draco_Mesh.hh
 * \author Ryan Wollaeger <wollaeger@lanl.gov>
 * \date   Thursday, Jun 07, 2018, 15:38 pm
 * \brief  Draco_Mesh class header file.
 * \note   Copyright (C) 2018-2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#ifndef rtt_mesh_Draco_Mesh_hh
#define rtt_mesh_Draco_Mesh_hh

#include "ds++/config.h"
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
 * Two important features for a fully realized Draco_Mesh are the following:
 * 1) Layout, which stores cell connectivity and hence the mesh topology.
 *    a) It has an internal layout containing local cell-to-cell linkage,
 *    b) and a boundary layout with side off-process linkage, and
 *    c) a ghost layout containing cell-to-ghostd-cell linkage.
 * 2) Geometry, which implies a metric for distance between points.
 *
 * Possibly temporary features:
 * 1) The cell_type_ vector (argument to the constructor) is currently taken to
 *    be the number of nodes per cell.
 * 2) The layout data structure(s) will probably be moved to a separate class,
 *    where accessors might be used on a flattened version.
 */
//===========================================================================//

class Draco_Mesh {
public:
  // >>> TYPEDEFS
  typedef rtt_mesh_element::Geometry Geometry;
  // \todo: update this to a full layout class?
  typedef std::map<unsigned,
                   std::vector<std::pair<unsigned, std::vector<unsigned>>>>
      Layout;

private:
  // >>> DATA

  // Dimension
  const unsigned dimension;

  // Geometry enumeration
  const Geometry geometry;

  // Number of cells
  const unsigned num_cells;

  // Number of nodes
  const unsigned num_nodes;

  // Side set flag (can be used for mapping BCs to sides)
  std::vector<unsigned> side_set_flag;

  // Ghost cell indices local to a different node, subscripted with a local
  // ghost cell index
  const std::vector<int> ghost_cell_number;

  // Node index for each ghost cell, subscripted with local ghost cell index
  const std::vector<int> ghost_cell_rank;

  // Vector subscripted with node index with coordinate vector
  const std::vector<std::vector<double>> node_coord_vec;

  // Cell types and node indices per cell
  const std::vector<unsigned> m_cell_type;
  const std::vector<unsigned> m_cell_to_node_linkage;

  // Side types and node indices per side
  std::vector<unsigned> m_side_node_count;
  std::vector<unsigned> m_side_to_node_linkage;

  // Layout of mesh: vector index is cell index, vector element is
  // description of cell's adjacency to other cells in the mesh.
  Layout cell_to_cell_linkage;

  // Side layout of mesh
  Layout cell_to_side_linkage;

  // Ghost cell layout of mesh
  Layout cell_to_ghost_cell_linkage;

public:
  //! Constructor.
  DLL_PUBLIC_mesh
  Draco_Mesh(unsigned dimension_, Geometry geometry_,
             const std::vector<unsigned> &cell_type_,
             const std::vector<unsigned> &cell_to_node_linkage_,
             const std::vector<unsigned> &side_set_flag_,
             const std::vector<unsigned> &side_node_count_,
             const std::vector<unsigned> &side_to_node_linkage_,
             const std::vector<double> &coordinates_,
             const std::vector<unsigned> &global_node_number_,
             const std::vector<unsigned> &ghost_cell_type_ = {},
             const std::vector<unsigned> &ghost_cell_to_node_linkage_ = {},
             const std::vector<int> &ghost_cell_number_ = {},
             const std::vector<int> &ghost_cell_rank_ = {});

  // >>> ACCESSORS

  unsigned get_dimension() const { return dimension; }
  Geometry get_geometry() const { return geometry; }
  unsigned get_num_cells() const { return num_cells; }
  unsigned get_num_nodes() const { return num_nodes; }
  const std::vector<unsigned> &get_side_set_flag() const {
    return side_set_flag;
  }
  const std::vector<int> &get_ghost_cell_numbers() const {
    return ghost_cell_number;
  }
  const std::vector<int> &get_ghost_cell_ranks() const {
    return ghost_cell_rank;
  }
  const std::vector<std::vector<double>> &get_node_coord_vec() const {
    return node_coord_vec;
  }
  const std::vector<unsigned> &get_cell_type() const { return m_cell_type; }
  const std::vector<unsigned> &get_cell_to_node_linkage() const {
    return m_cell_to_node_linkage;
  }
  const std::vector<unsigned> &get_side_node_count() const {
    return m_side_node_count;
  }
  const std::vector<unsigned> &get_side_to_node_linkage() const {
    return m_side_to_node_linkage;
  }
  const Layout &get_cc_linkage() const { return cell_to_cell_linkage; }
  const Layout &get_cs_linkage() const { return cell_to_side_linkage; }
  const Layout &get_cg_linkage() const { return cell_to_ghost_cell_linkage; }

  // >>> SERVICES

private:
  // >>> SUPPORT FUNCTIONS

  //! Calculate (merely de-serialize) the vector of node coordinates
  std::vector<std::vector<double>>
  compute_node_coord_vec(const std::vector<double> &coordinates) const;

  // \todo: add layout class and complete temporary version of this function.
  //! Calculate the cell-to-cell linkage (layout)
  void compute_cell_to_cell_linkage(
      const std::vector<unsigned> &cell_type,
      const std::vector<unsigned> &cell_to_node_linkage,
      const std::vector<unsigned> &side_node_count,
      const std::vector<unsigned> &side_to_node_linkage,
      const std::vector<unsigned> &ghost_cell_type,
      const std::vector<unsigned> &ghost_cell_to_node_linkage);

  //! Calculate a map of node to vectors of indices (cells, sides, ghost cells)
  std::map<unsigned, std::vector<unsigned>> compute_node_indx_map(
      const std::vector<unsigned> &indx_type,
      const std::vector<unsigned> &indx_to_node_linkage) const;
};

} // end namespace rtt_mesh

#endif // rtt_mesh_Draco_Mesh_hh

//---------------------------------------------------------------------------//
// end of mesh/Draco_Mesh.hh
//---------------------------------------------------------------------------//
