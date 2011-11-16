//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   cdi_ipcress/IpcressMultigroupOpacity.cc
 * \author Kelly Thompson
 * \date   Tue Nov 15 15:51:27 2011
 * \brief  IpcressMultigroupOpacity templated class implementation file.
 * \note   Copyright (C) 2011 Los Alamos National Security, LLC.
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#include "IpcressMultigroupOpacity.hh"
#include "IpcressFile.hh"       // we have smart pointers to
                                // IpcressFile objects.
#include "IpcressDataTable.hh"  // we have a smart pointer to a
                                // IpcressDataTable object.

#include "ds++/Assert.hh" // we make use of Require()
#include "ds++/Packing_Utils.hh"
#include <cmath> // we need to define log(double) and exp(double)


namespace rtt_cdi_ipcress
{
    
// ------------ //
// Constructors //
// ------------ //
    
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor for IpcressMultigroupOpacity object.
 * 
 * See IpcressMultigroupOpacity.hh for details.
 */
IpcressMultigroupOpacity::IpcressMultigroupOpacity( 
    rtt_dsxx::SP< IpcressFile const > const & in_spIpcressFile,
    size_t            in_materialID,
    rtt_cdi::Model    in_opacityModel,
    rtt_cdi::Reaction in_opacityReaction )
    : spIpcressFile( in_spIpcressFile ),
      materialID( in_materialID ),
      numKeys( 0 ),
      vKnownKeys(),
      opacityModel( in_opacityModel ),
      opacityReaction( in_opacityReaction ),
      energyPolicyDescriptor( "mg" ),
      spIpcressDataTable()
{
    // Verify that the requested material ID is available in the specified
    // IPCRESS file.
    Insist( spIpcressFile->materialFound( materialID ),
            std::string("The requested material ID is not available in the ") +
            std::string("specified Ipcress file.") );
	    
    // Retrieve keys available for this material from the IPCRESS file.
    vKnownKeys = spIpcressFile->listDataFieldNames( materialID );
    Check(vKnownKeys.size()>0);
	    
    // Create the data table object and fill it with the table
    // data from the IPCRESS file.
    spIpcressDataTable = new IpcressDataTable(
	energyPolicyDescriptor,
        opacityModel,
        opacityReaction,
	vKnownKeys,
        materialID,
        spIpcressFile );
	    
} // end of IpcressData constructor

//---------------------------------------------------------------------------//
/*!
 * \brief Unpacking constructor for IpcressMultigroupOpacity object.
 * 
 * See IpcressMultigroupOpacity.hh for details.
 */
IpcressMultigroupOpacity::IpcressMultigroupOpacity(
    std::vector<char> const & packed )
    : spIpcressFile(),
      materialID(0),
      numKeys(0),
      vKnownKeys(),
      opacityModel(),
      opacityReaction(),
      energyPolicyDescriptor( "mg" ),
      spIpcressDataTable()
{
    Require (packed.size() >= 5 * sizeof(int));

    // make an unpacker
    rtt_dsxx::Unpacker unpacker;
    unpacker.set_buffer(packed.size(), &packed[0]);

    // unpack and check the descriptor
    int         packed_descriptor_size = 0;
    unpacker >> packed_descriptor_size;
    Check (packed_descriptor_size > 0);

    // make a vector<char> for the packed descriptor
    std::vector<char> packed_descriptor(packed_descriptor_size);

    // unpack it
    std::string descriptor;
    for (size_t i = 0; i < static_cast<size_t>(packed_descriptor_size); i++)
	unpacker >> packed_descriptor[i];
    rtt_dsxx::unpack_data(descriptor, packed_descriptor);

    // make sure it is "gray"
    Insist (descriptor == "mg", 
	    "Tried to unpack a non-mg opacity in IpcressMultigroupOpacity.");

    // unpack the size of the packed filename
    int packed_filename_size(0);
    unpacker >> packed_filename_size;

    // make a vector<char> for the packed filename
    std::vector<char> packed_filename(packed_filename_size);

    // unpack it
    std::string filename;
    for (size_t i = 0; i < static_cast<size_t>(packed_filename_size); i++)
	unpacker >> packed_filename[i];
    rtt_dsxx::unpack_data(filename, packed_filename);

    // unpack the material id
    int itmp(0);
    unpacker >> itmp;
    materialID = static_cast<size_t>(itmp);

    // unpack the model and reaction
    int model    = 0;
    int reaction = 0;
    unpacker >> model >> reaction;

    opacityModel    = static_cast<rtt_cdi::Model>(model);
    opacityReaction = static_cast<rtt_cdi::Reaction>(reaction);

    Ensure (unpacker.get_ptr() == &packed[0] + packed.size());
    
    // build a new IpcressFile
    spIpcressFile = new IpcressFile(filename);
    Check (spIpcressFile);

    // Verify that the requested material ID is available in the specified
    // IPCRESS file.
    Insist( spIpcressFile->materialFound( materialID ),
        "Requested material ID is not found in the specified Ipcress file.");
            
    // Retrieve keys available fo this material from the IPCRESS file.
    vKnownKeys = spIpcressFile->listDataFieldNames( materialID );
    Check(vKnownKeys.size()>0);
    
    // Create the data table object and fill it with the table
    // data from the IPCRESS file.
    spIpcressDataTable = new IpcressDataTable(
	energyPolicyDescriptor,
        opacityModel,
        opacityReaction,
	vKnownKeys,
        materialID,
        spIpcressFile );

    Ensure (spIpcressFile);
    Ensure (spIpcressDataTable);
} 
    
/*!
 * \brief Default IpcressOpacity() destructor.
 *
 * \sa This is required to correctly release memory when a 
 *     IpcressMultigroupOpacity is destroyed.  This constructor's
 *     * definition must be declared in the implementation file so
 *     that * we can avoid including too many header files
 */
IpcressMultigroupOpacity::~IpcressMultigroupOpacity()
{
    // empty
}
    
    
// --------- //
// Accessors //
// --------- //
    
/*!
 * \brief Returns a "plain English" description of the opacity
 *     data that this class references. (e.g. "Multigroup Rosseland
 *     Scattering".) 
 *
 * The definition of this function is not included here to prevent the
 * inclusion of the IpcressFile.hh definitions within this header file.
 */
std::string IpcressMultigroupOpacity::getDataDescriptor() const 
{
    // call the correct function from the IpcressDataTable object.
    return spIpcressDataTable->getDataDescriptor(); 
}
    
/*!
 * \brief Returns the name of the associated IPCRESS file.
 *
 * The definition of this function is not included here to prevent the
 * inclusion of the IpcressFile.hh definitions within this header file.
 */
std::string IpcressMultigroupOpacity::getDataFilename() const 
{
    return spIpcressFile->getDataFilename(); 
}
    
/*!
 * \brief Opacity accessor that returns a single opacity (or a vector of
 *     opacities for the multigroup EnergyPolicy) that corresponds to the
 *     provided temperature and density.
 */
std::vector< double > IpcressMultigroupOpacity::getOpacity(
    double targetTemperature,
    double targetDensity ) const
{ 
    // number of groups in this multigroup set.
    size_t const numGroups = spIpcressDataTable->getNumGroupBoundaries() - 1;
	    
    // temporary opacity vector used by the wrapper.  The returned data will
    // be copied into the opacityIterator.
    std::vector<double> opacity( numGroups, -99.0 );
    
    // logarithmic interpolation:
    for( size_t g=0; g<numGroups; ++g )
    {
        opacity[g] = spIpcressDataTable->interpOpac( targetTemperature,
                                                     targetDensity, g );
        Check( opacity[g] >= 0.0 );
    }
    return opacity;
}
    
/*!
 * \brief Opacity accessor that returns a vector of opacities (or a
 *     vector of vectors of opacities for the multigroup
 *     EnergyPolicy) that correspond to the provided vector of
 *     temperatures and a single density value.
 */
std::vector< std::vector< double > > IpcressMultigroupOpacity::getOpacity(
    std::vector<double> const & targetTemperature,
    double targetDensity ) const
{ 
    std::vector< std::vector< double > > opacity( targetTemperature.size() );
    for ( size_t i=0; i<targetTemperature.size(); ++i )
        opacity[i] = getOpacity( targetTemperature[i], targetDensity );
    return opacity;
}
    
/*!
 * \brief Opacity accessor that returns a vector of opacities (or a
 *     vector of vectors of opacities for the multigroup
 *     EnergyPolicy) that correspond to the provided vector of
 *     densities and a single temperature value.
 */
std::vector< std::vector< double > > IpcressMultigroupOpacity::getOpacity(
    double targetTemperature,
    const std::vector<double>& targetDensity ) const
{ 
    std::vector< std::vector< double > > opacity( targetDensity.size() );
    for ( size_t i=0; i<targetDensity.size(); ++i )
        opacity[i] = getOpacity( targetTemperature, targetDensity[i] );
    return opacity;
}
    
/*!
 * \brief Returns a vector of temperatures that define the cached
 *     opacity data table.
 */
std::vector< double > IpcressMultigroupOpacity::getTemperatureGrid() const
{
    return spIpcressDataTable->getTemperatures();
}
    
/*!
 * \brief Returns the size of the temperature grid.
 */
size_t IpcressMultigroupOpacity::getNumTemperatures() const
{
    return spIpcressDataTable->getNumTemperatures();
}
    
/*!
 * \brief Returns a vector of densities that define the cached
 *     opacity data table.
 */
std::vector<double> IpcressMultigroupOpacity::getDensityGrid() const
{
    return spIpcressDataTable->getDensities();
}
    
/*! 
 * \brief Returns the size of the density grid.
 */
size_t IpcressMultigroupOpacity::getNumDensities() const
{
    return spIpcressDataTable->getNumDensities();
}
    
/*!
 * \brief Returns a vector of energy values (keV) that define the
 *     energy boundaries of the cached multigroup opacity data
 *     table.  (This accessor is only valid for the Multigroup
 *     EnergyPolicy version of IpcressMultigroupOpacity.)
 */
std::vector< double > IpcressMultigroupOpacity::getGroupBoundaries() const
{
    return spIpcressDataTable->getGroupBoundaries();
}
    
/*!
 * \brief Returns the number of group boundaries found in the
 *     current multigroup data set.
 */
size_t IpcressMultigroupOpacity::getNumGroupBoundaries() const
{
    return spIpcressDataTable->getNumGroupBoundaries();
}

// ------- //
// Packing //
// ------- //

/*!
 * Pack the IpcressMultigroupOpacity state into a char string represented by
 * a vector<char>. This can be used for persistence, communication, etc. by
 * accessing the char * under the vector (required by implication by the
 * standard) with the syntax &char_string[0]. Note, it is unsafe to use
 * iterators because they are \b not required to be char *.
 */
std::vector<char> IpcressMultigroupOpacity::pack() const
{
    using std::vector;
    using std::string;

    Require (spIpcressFile);

    // pack up the energy policy descriptor
    vector<char> packed_descriptor;
    rtt_dsxx::pack_data(energyPolicyDescriptor, packed_descriptor);

    // pack up the ipcress file name
    string       filename = spIpcressFile->getDataFilename();
    vector<char> packed_filename;
    rtt_dsxx::pack_data(filename, packed_filename);

    // determine the total size: 3 ints (reaction, model, material id) + 2
    // ints for packed_filename size and packed_descriptor size + char in
    // packed_filename and packed_descriptor
    size_t size = 5 * sizeof(int) + packed_filename.size() + 
	packed_descriptor.size();

    // make a container to hold packed data
    vector<char> packed(size);

    // make a packer and set it
    rtt_dsxx::Packer packer;
    packer.set_buffer(size, &packed[0]);

    // pack the descriptor
    packer << static_cast<int>(packed_descriptor.size());
    for (size_t i = 0; i < packed_descriptor.size(); i++)
	packer << packed_descriptor[i];

    // pack the filename (size and elements)
    packer << static_cast<int>(packed_filename.size());
    for (size_t i = 0; i < packed_filename.size(); i++)
	packer << packed_filename[i];

    // pack the material id
    packer << static_cast<int>(materialID);

    // pack the model and reaction
    packer << static_cast<int>(opacityModel) 
	   << static_cast<int>(opacityReaction);

    Ensure (packer.get_ptr() == &packed[0] + size);
    return packed;
}
    
} // end namespace rtt_cdi_ipcress

//---------------------------------------------------------------------------//
// end of IpcressMultigroupOpacity.cc
//---------------------------------------------------------------------------//