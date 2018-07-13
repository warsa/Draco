//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/test/tstX3D_Draco_Mesh_Reader.cc
 * \author Ryan Wollaeger <wollaeger@lanl.gov>
 * \date   Tuesday, Jul 10, 2018, 10:23 am
 * \brief  X3D_Draco_Mesh_Reader class unit test.
 * \note   Copyright (C) 2018 Los Alamos National Security, LLC.
 *         All rights reserved. */
//---------------------------------------------------------------------------//

#include "mesh/X3D_Draco_Mesh_Reader.hh"
#include "c4/ParallelUnitTest.hh"
#include "ds++/DracoStrings.hh"
#include "ds++/Release.hh"
#include "ds++/Soft_Equivalence.hh"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>

using rtt_mesh::X3D_Draco_Mesh_Reader;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

// Parse an X3D file format and compare to reference
void read_x3d_mesh_2d(rtt_c4::ParallelUnitTest &ut) {

  // >>> PARSE MESH

  const std::string inputpath = ut.getTestSourcePath();
  const std::string filename = inputpath + "x3d.mesh.in";

  // construct reader
  std::shared_ptr<X3D_Draco_Mesh_Reader> x3d_reader(
      new X3D_Draco_Mesh_Reader(filename));

  // read mesh
  x3d_reader->read_mesh();

  // >>> CHECK HEADER DATA

  if (x3d_reader->get_process() != 0)
    ITFAILS;

  if (x3d_reader->get_numdim() != 2)
    ITFAILS;

  if (x3d_reader->get_numcells() != 1)
    ITFAILS;

  if (x3d_reader->get_numnodes() != 4)
    ITFAILS;

  // >>> CHECK CELL-NODE DATA

  if (x3d_reader->get_celltype(0) != 4)
    ITFAILS;

  std::vector<int> test_cellnodes = {0, 1, 2, 3};
  if (x3d_reader->get_cellnodes(0) != test_cellnodes)
    ITFAILS;

  // >>> CHECK NODE-COORD DATA

  std::vector<std::vector<double>> test_coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
  for (int node = 0; node < 4; ++node) {
    if (x3d_reader->get_nodecoord(node) != test_coords[node])
      ITFAILS;
  }

  // >>> CHECK SIDE DATA

  // \todo: this should change when boundary file parsing is enabled
  if (x3d_reader->get_numsides() != 0)
    ITFAILS;
  if (x3d_reader->get_sidetype(0) != 0)
    ITFAILS;
  if (x3d_reader->get_sideflag(0) != 0)
    ITFAILS;

  // successful test output
  if (ut.numFails == 0)
    PASSMSG("2D X3D_Draco_Mesh_Reader tests ok.");
  return;
}

//---------------------------------------------------------------------------//

int main(int argc, char *argv[]) {
  rtt_c4::ParallelUnitTest ut(argc, argv, rtt_dsxx::release);
  try {
    Insist(rtt_c4::nodes() == 1, "This test only uses 1 PE.");
    read_x3d_mesh_2d(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------//
// end of mesh/test/tstX3D_Draco_Mesh_Reader.cc
//---------------------------------------------------------------------------//
