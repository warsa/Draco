#include "Compton_NWA.hh"
#include "compton_file.hh"
#include "multigroup_compton_data.hh"
#include "multigroup_lib_builder.hh"

namespace rtt_compton {

Compton_NWA::Compton_NWA(const std::string &filehandle) {

  // Make a compton file object
  compton_file Cfile(false);

  std::shared_ptr<multigroup_compton_data> Cdata =
      Cfile.read_mg_csk_data(filehandle);
}
}
