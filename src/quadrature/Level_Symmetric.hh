//----------------------------------*-C++-*----------------------------------------------//
/*!
 * \file   quadrature/Level_Symmetric.hh
 * \author Kelly Thompson
 * \date   Wed Sep  1 10:19:52 2004
 * \brief  A class to encapsulate a 3D Level Symmetric quadrature set.
 * \note   Copyright 2004 The Regents of the University of California.
 *
 * Long description.
 */
//---------------------------------------------------------------------------------------//
// $Id: Level_Symmetric.hh 6718 2012-08-30 20:03:01Z warsa $
//---------------------------------------------------------------------------------------//

#ifndef quadrature_Level_Symmetric_hh
#define quadrature_Level_Symmetric_hh

#include "Octant_Quadrature.hh"

namespace rtt_quadrature
{

//=======================================================================================//
/*!
 * \class Level_Symmetric
 * \brief A class to encapsulate a 3D Level Symmetric quadrature set.
 */
//=======================================================================================//

class Level_Symmetric : public Octant_Quadrature
{
  public:

    // CREATORS

    // The default values for snOrder_ and norm_ were set in QuadCreator.
    explicit Level_Symmetric( unsigned sn_order,
                              QIM const qim = SN)
        :
        Octant_Quadrature(qim),
        sn_order_( sn_order)
        
    {
        Require(sn_order>0 && sn_order%2==0);
    }

    Level_Symmetric();    // disable default construction

    // ACCESSORS
    
    unsigned sn_order()     const { return sn_order_; }

    // SERVICES
    
    // These functions override the virtual member functions specifed in the
    // parent class Quadrature.

    string name()        const;
    
    string parse_name()  const;
    
    unsigned number_of_levels() const;
    
    string as_text(string const &indent) const;
    
    // STATICS

    static SP<Quadrature> parse(Token_Stream &tokens);

  private:

    // IMPLEMENTATION

    //! Virtual hook for create_ordinate_set
    virtual void create_octant_ordinates_(vector<double> &mu,
                                          vector<double> &eta,
                                          vector<double> &wt) const;

    // DATA
    unsigned sn_order_;
};

} // end namespace rtt_quadrature

#endif // quadrature_Level_Symmetric_hh

//---------------------------------------------------------------------------------------//
//              end of quadrature/Level_Symmetric.hh
//---------------------------------------------------------------------------------------//