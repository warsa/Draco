//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/test/tstDraco_Mesh.cc
 * \date   May 2018
 * \brief  Draco_Mesh class unit test.
 * \note   Copyright (C) 2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "mesh/Draco_Mesh.hh"
#include "c4/ParallelUnitTest.hh"
#include "ds++/Release.hh"
#include <bitset>

using rtt_mesh::Draco_Mesh;

// 2D Cartesian mesh construction test
void cartesian_mesh_2d(rtt_c4::ParallelUnitTest &ut) {

  // instantiate a mesh, to be defined below
  std::shared_ptr<Draco_Mesh> mesh;

  //>>> SET UP CELL AND NODE DATA
  {
    // TODO: parse mesh data instead of hard-coding
    // TODO: make mesh data setup a builder and move to separate file.

    // set the number of cells and nodes
    const size_t num_xdir = 1;
    const size_t num_ydir = 1;
    const size_t num_cells = num_xdir * num_ydir;
    const size_t num_nodes = (num_xdir + 1) * (num_ydir + 1);

    // use two dimensions and Cartesian geometry
    const unsigned dim = 2;
    const Draco_Mesh::Geometry geometry = Draco_Mesh::Geometry::CARTESIAN;

    // set cell type (quadrilateral cells)
    std::vector<unsigned> cell_type(num_cells, 4);

    // set the cell-to-node linkage counterclockwise about each cell
    std::vector<unsigned> cell_to_node_linkage(num_cells * 4);
    for (size_t j = 0; j < num_ydir; ++j) {
      for (size_t i = 0; i < num_xdir; ++i) {

        // get a serial cell index
        size_t cell = i + num_xdir * j;

        // set each node entry per cell
        cell_to_node_linkage[4 * cell] = cell + j;
        cell_to_node_linkage[4 * cell + 1] = cell + j + 1;
        cell_to_node_linkage[4 * cell + 2] = cell + num_xdir + 1 + j + 1;
        cell_to_node_linkage[4 * cell + 3] = cell + num_xdir + 1 + j;
      }
    }

    // set the number of sides and side flags
    size_t poff = num_xdir + num_ydir; // parallel side offset
    size_t num_sides = 2 * poff;

    // set two nodes per each side cell (only possibility in 2D)
    std::vector<unsigned> side_node_count(num_sides, 2);

    // calculate arrays storing data about the sides of the mesh
    std::vector<unsigned> side_set_flag(num_sides);
    std::vector<unsigned> side_to_node_linkage(2 * num_sides);
    // ... over top and bottom faces
    for (size_t i = 0; i < num_xdir; ++i) {
      // bottom face
      side_set_flag[i] = 1;
      side_to_node_linkage[2 * i] = i;
      side_to_node_linkage[2 * i + 1] = i + 1;
      // top face
      side_set_flag[i + poff] = 3;
      side_to_node_linkage[2 * (i + poff)] = num_nodes - 1 - i;
      side_to_node_linkage[2 * (i + poff) + 1] = num_nodes - 1 - i - 1;
    }
    // ... over left and right faces
    for (size_t j = 0; j < num_ydir; ++j) {
      // right face
      side_set_flag[j + num_xdir] = 2;
      side_to_node_linkage[2 * (j + num_xdir)] = num_xdir * (j + 1) + j;
      side_to_node_linkage[2 * (j + num_xdir) + 1] = num_xdir * (j + 2) + j + 1;
      // left face
      side_set_flag[j + poff + num_xdir] = 4;
      side_to_node_linkage[2 * (j + poff + num_xdir)] =
          num_nodes - 1 - num_xdir * (j + 1) - j;
      side_to_node_linkage[2 * (j + poff + num_xdir) + 1] =
          num_nodes - 1 - num_xdir * (j + 2) - (j + 1);
    }

    // set some coordinates and global node indices
    std::vector<double> coordinates(dim * num_nodes);
    std::vector<unsigned> global_node_number(num_nodes);
    for (unsigned i = 0; i < num_nodes; ++i) {

      // TODO: generalize coordinate generation
      std::bitset<8> b2(i);
      coordinates[dim * i] = static_cast<double>(b2[0]);
      coordinates[1 + dim * i] = static_cast<double>(b2[1]);

      // TODO: generalize for multi-processing
      global_node_number[i] = i;
    }

    // set global node indices

    // TODO: eventually remove these print statements
    for (size_t j = 0; j < num_sides; ++j) {
      std::cout << "side_set_flag[" << j << "] = " << side_set_flag[j]
                << std::endl;
      std::cout << std::endl;
      std::cout << "side_to_node_linkage[2 * " << j
                << "] = " << side_to_node_linkage[2 * j] << std::endl;
      std::cout << "side_to_node_linkage[2 * " << j
                << " + 1] = " << side_to_node_linkage[2 * j + 1] << std::endl;
      std::cout << std::endl;
    }

    // TODO: eventually remove these print statements
    for (size_t i = 0; i < num_nodes; ++i) {
      std::cout << "coordinates[dim * " << i << "] = " << coordinates[dim * i]
                << std::endl;
      std::cout << "coordinates[dim * " << i + 1
                << "] = " << coordinates[1 + dim * i] << std::endl;
    }

    // build the mesh
    mesh.reset(new Draco_Mesh(dim, geometry, cell_type, cell_to_node_linkage,
                              side_set_flag, side_node_count,
                              side_to_node_linkage, coordinates,
                              global_node_number));
  }

  // successful test output
  if (ut.numFails == 0)
    PASSMSG("2D Draco_Mesh tests ok.");
  return;
}

//---------------------------------------------------------------------------//

int main(int argc, char *argv[]) {
  rtt_c4::ParallelUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    Insist(rtt_c4::nodes() == 1, "This test only uses 1 PE.");
    cartesian_mesh_2d(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
// end of tstDraco_Mesh.cc
//---------------------------------------------------------------------------//
