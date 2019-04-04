//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/test/tstDraco_Mesh.cc
 * \author Ryan Wollaeger <wollaeger@lanl.gov>
 * \date   Thursday, Jun 07, 2018, 15:43 pm
 * \brief  Draco_Mesh class unit test.
 * \note   Copyright (C) 2018-2019 Triad National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "Test_Mesh_Interface.hh"
#include "c4/ParallelUnitTest.hh"
#include "ds++/Release.hh"

using rtt_mesh::Draco_Mesh;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

// 2D Cartesian mesh construction test
void cartesian_mesh_2d(rtt_c4::ParallelUnitTest &ut) {

  // TODO: parse mesh data instead of hard-coding

  // use Cartesian geometry
  const Draco_Mesh::Geometry geometry = Draco_Mesh::Geometry::CARTESIAN;

  //>>> SET UP CELL AND NODE DATA

  // set the number of cells and nodes
  const size_t num_xdir = 2;
  const size_t num_ydir = 1;

  // generate a constainer for data needed in mesh construction
  rtt_mesh_test::Test_Mesh_Interface mesh_iface(num_xdir, num_ydir);

  // short-cut to some arrays
  const std::vector<unsigned> &cell_type = mesh_iface.cell_type;
  const std::vector<unsigned> &cell_to_node_linkage =
      mesh_iface.cell_to_node_linkage;
  const std::vector<unsigned> &side_node_count = mesh_iface.side_node_count;
  const std::vector<unsigned> &side_to_node_linkage =
      mesh_iface.side_to_node_linkage;

  // instantiate the mesh
  std::shared_ptr<Draco_Mesh> mesh(new Draco_Mesh(
      mesh_iface.dim, geometry, cell_type, cell_to_node_linkage,
      mesh_iface.side_set_flag, side_node_count, side_to_node_linkage,
      mesh_iface.coordinates, mesh_iface.global_node_number));

  // check that the scalar data is correct
  FAIL_IF_NOT(mesh->get_dimension() == 2);
  FAIL_IF_NOT(mesh->get_geometry() == Draco_Mesh::Geometry::CARTESIAN);
  FAIL_IF_NOT(mesh->get_num_cells() == mesh_iface.num_cells);
  FAIL_IF_NOT(mesh->get_num_nodes() == mesh_iface.num_nodes);

  // check that flat cell type and cell to node linkage is correct
  FAIL_IF_NOT(mesh->get_cell_type() == cell_type);
  FAIL_IF_NOT(mesh->get_cell_to_node_linkage() == cell_to_node_linkage);

  // check that flat side type and side to node linkage is correct
  FAIL_IF_NOT(mesh->get_side_node_count() == side_node_count);
  FAIL_IF_NOT(mesh->get_side_to_node_linkage() == side_to_node_linkage);
  FAIL_IF_NOT(mesh->get_side_set_flag() == mesh_iface.side_set_flag);

  // get the layout generated by the mesh
  const Draco_Mesh::Layout &layout = mesh->get_cc_linkage();

  // check that the layout has been generated
  FAIL_IF_NOT(layout.size() == mesh_iface.num_cells);

  // check that each cell has the correct neighbors
  {
    std::map<unsigned, std::vector<unsigned>> test_cell_map;
    for (unsigned j = 0; j < num_ydir; ++j) {
      for (unsigned i = 0; i < num_xdir; ++i) {

        // calculate the cell index
        unsigned cell = i + j * num_xdir;

        // calculate neighbor cell indices
        if (i > 0)
          test_cell_map[cell].push_back(cell - 1);
        if (i < num_xdir - 1)
          test_cell_map[cell].push_back(cell + 1);
        if (j > 0)
          test_cell_map[cell].push_back(cell - num_xdir);
        //if (j < num_ydir - 1)
        //  test_cell_map[cell].push_back(cell + num_xdir);
      }
    }

    for (unsigned cell = 0; cell < mesh_iface.num_cells; ++cell) {

      // get number of faces per cell in layout
      const size_t num_faces = layout.at(cell).size();

      // check that the number of faces per cell is correct
      FAIL_IF_NOT(num_faces == test_cell_map[cell].size());

      // check that cell neighbors are correct
      for (unsigned face = 0; face < num_faces; ++face)
        FAIL_IF_NOT(layout.at(cell)[face].first == test_cell_map[cell][face]);
    }
  }

  // get the boundary layout generated by the mesh
  const Draco_Mesh::Layout &bd_layout = mesh->get_cs_linkage();

  // check that the boundary (or side) layout has been generated
  FAIL_IF_NOT(bd_layout.size() == mesh_iface.num_cells);

  // get the ghost-cell layout generated by the mesh
  const Draco_Mesh::Layout &go_layout = mesh->get_cg_linkage();

  // check that there are no ghost cells for this mesh
  FAIL_IF_NOT(go_layout.size() == 0);

  // check that cell-to-node linkage data is correct
  {
    std::vector<unsigned> test_cn_linkage =
        mesh_iface.flatten_cn_linkage(layout, bd_layout, go_layout);

    // check that cn_linkage is a permutation of the original cell-node linkage
    std::vector<unsigned>::const_iterator cn_first =
        cell_to_node_linkage.begin();
    std::vector<unsigned>::const_iterator test_cn_first =
        test_cn_linkage.begin();
    for (unsigned cell = 0; cell < mesh_iface.num_cells; ++cell) {

      // nodes must only be permuted at the cell level
      FAIL_IF_NOT(std::is_permutation(test_cn_first,
                                      test_cn_first + cell_type[cell], cn_first,
                                      cn_first + cell_type[cell]));

      // update the iterators
      cn_first += cell_type[cell];
      test_cn_first += cell_type[cell];
    }
  }

  // check that each cell has the correct sides
  {
    std::vector<unsigned> test_sn_linkage =
        mesh_iface.flatten_sn_linkage(bd_layout);

    // check that sn_linkage is a permutation of the original side-node linkage
    std::vector<unsigned>::const_iterator sn_first =
        side_to_node_linkage.begin();
    std::vector<unsigned>::const_iterator test_sn_first =
        test_sn_linkage.begin();
    for (unsigned side = 0; side < mesh_iface.num_sides; ++side) {

      // sn_linkage must be a permutation of the original side-node linkage
      FAIL_IF_NOT(std::is_permutation(
          test_sn_first, test_sn_first + side_node_count[side], sn_first,
          sn_first + side_node_count[side]));

      // update the iterators
      sn_first += side_node_count[side];
      test_sn_first += side_node_count[side];
    }
  }

  // test default side (boundary) data
  {
    // set reference default side-set flags
    std::vector<unsigned> nobc_side_set_flag(mesh_iface.num_sides, 0);

    // instantiate a version of the mesh without side (b.c.) data
    std::shared_ptr<Draco_Mesh> mesh_no_bc_data(new Draco_Mesh(
        mesh_iface.dim, geometry, cell_type, cell_to_node_linkage, {}, {}, {},
        mesh_iface.coordinates, mesh_iface.global_node_number));

    // check that default data has been initialized at mesh boundaries
    FAIL_IF_NOT(mesh_no_bc_data->get_side_node_count() == side_node_count);
    const std::vector<unsigned> &nobc_sn_linkage =
        mesh_no_bc_data->get_side_to_node_linkage();
    FAIL_IF_NOT(std::is_permutation(nobc_sn_linkage.begin(),
                                    nobc_sn_linkage.end(),
                                    side_to_node_linkage.begin()));
    FAIL_IF_NOT(mesh_no_bc_data->get_side_set_flag() == nobc_side_set_flag);
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
// end of mesh/test/tstDraco_Mesh.cc
//---------------------------------------------------------------------------//
