//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   parser/test/tstClass_Parse_Table.cc
 * \brief  Unit tests for the Class_Parse_Table template.
 * \note   Copyright (C) 2016-2018 TRIAD, LLC. All rights reserved.
  */
//---------------------------------------------------------------------------//

#include "ds++/Release.hh"
#include "ds++/ScalarUnitTest.hh"
#include "parser/Class_Parser.hh"
#include "parser/File_Token_Stream.hh"
#include "parser/String_Token_Stream.hh"
#include "parser/utilities.hh"
#include <string.h>

#ifdef _MSC_VER
#undef ERROR
#endif

using namespace std;
using namespace rtt_parser;
using namespace rtt_dsxx;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// A model class parameter.

enum Color { BLACK, BLUE, BLUE_GREEN, SENTINEL };

char const *color_table[] = {"BLACK", "BLUE", "BLUE_GREEN"};

//---------------------------------------------------------------------------//
// Model class for which we want to parse specifications

class Vehicle {
public:
  explicit Vehicle(Color const color) : color_(color) {}

  Color color() const noexcept { return color_; }

private:
  Color color_;
};

//---------------------------------------------------------------------------//
// Specialization of Class_Parser for our model class.

template <>
class rtt_parser::Class_Parser<Vehicle> : public Class_Parser_Base<Vehicle> {
public:
  Class_Parser();
  // context argument is optional

  void check_completeness(Token_Stream &ut) {
    ut.check_semantics(color != SENTINEL, "no color specified");
  }

  Vehicle *create_object() { return new Vehicle(color); }

protected:
  // Data to be parsed which will be used to construct the class, for example:

  Color color;

  //private:
public: // To facilitate keyword invariant testing
  // Parse functions

  void parse_color(Token_Stream &, int);
  void parse_any_color(Token_Stream &, int);

  // Raw parse table

  static Class_Parser_Keyword<Vehicle> const raw_table[];
};

//---------------------------------------------------------------------------//
// Raw keyword table for our Class_Parser specialization.

Class_Parser_Keyword<Vehicle> const Class_Parser<Vehicle>::raw_table[] = {
    {"BLUE", &rtt_parser::Class_Parser<Vehicle>::parse_color, 1, "main"},
    {"BLACK", &rtt_parser::Class_Parser<Vehicle>::parse_color, 0, "main"},
    {"BLUE GREEN", &rtt_parser::Class_Parser<Vehicle>::parse_color, 2, "main"},
    {"BLUISH GREEN", &rtt_parser::Class_Parser<Vehicle>::parse_color, 2, "main",
     "alternate keyword for same parmeter is permitted"},
    {"color", &rtt_parser::Class_Parser<Vehicle>::parse_any_color, 0, "main"},
};

//---------------------------------------------------------------------------//
// Mandatory constructor for our Class_Parser specialization.
rtt_parser::Class_Parser<Vehicle>::Class_Parser()
    : Class_Parser_Base<Vehicle>(*this, raw_table,
                                 sizeof(raw_table) / sizeof(raw_table[0])),
      color(SENTINEL) {}

//---------------------------------------------------------------------------//
// Optional parse functions for our Class_Parser specialization.

void rtt_parser::Class_Parser<Vehicle>::parse_color(Token_Stream &tokens,
                                                    int i) {
  Require(i < SENTINEL);
  tokens.check_semantics(color == SENTINEL, "duplicate color specification");
  color = (Color)i;
}

void rtt_parser::Class_Parser<Vehicle>::parse_any_color(Token_Stream &tokens,
                                                        int) {
  Token token = tokens.shift();
  for (unsigned i = 0; i < sizeof(color_table) / sizeof(const char *); i++)
    if (!strcmp(token.text().c_str(), color_table[i])) {
      color = (Color)i;
      return;
    }

  tokens.report_syntax_error(token, "expected a color");
}

//---------------------------------------------------------------------------//
// class_parse specialization for our model class.

template <>
shared_ptr<Vehicle> rtt_parser::parse_class<Vehicle>(Token_Stream &tokens) {
  rtt_parser::Class_Parser<Vehicle> parser;
  return parser.parse(tokens);
}

//---------------------------------------------------------------------------//
// Now a model class that happens to have no parameters. While this may seem
// pointless, we allow parsers for empty classes as placeholders.

class Empty {};

template <>
class rtt_parser::Class_Parser<Empty>
    : public rtt_parser::Class_Parser_Base<Empty, false, true> {
public:
  Class_Parser() : Class_Parser_Base<Empty, false, true>(*this) {}

protected:
  // IMPLEMENTATION

  void check_completeness(Token_Stream &){};

  Empty *create_object() { return new Empty(); }
};

template <>
shared_ptr<Empty> rtt_parser::parse_class<Empty>(Token_Stream &tokens) {
  rtt_parser::Class_Parser<Empty> parser;
  return parser.parse(tokens);
}

//---------------------------------------------------------------------------//
// Inheritance.

//------- Father
class Car : public Vehicle {
public:
  explicit Car(Color color) : Vehicle(color) {}
};

template <>
class rtt_parser::Class_Parser<Car> : public Class_Parser<Vehicle>,
                                      public Class_Parser_Base<Car> {
public:
  typedef Class_Parser_Base<Car> Base;

  Class_Parser();
};

rtt_parser::Class_Parser<Car>::Class_Parser() : Class_Parser_Base<Car>(*this) {
  static bool first_time = true;
  if (first_time) {
    Base::add(Class_Parser_Base<Vehicle>::table());
    first_time = false;
  }
}

//------- Child
class Pickup : public Car {
public:
  explicit Pickup(Color color) : Car(color) {}
};

template <>
class rtt_parser::Class_Parser<Pickup> : public Class_Parser<Car>,
                                         public Class_Parser_Base<Pickup> {
public:
  typedef Class_Parser_Base<Pickup> Base;

  Class_Parser();

  void check_completeness(Token_Stream &) {
    // No requirement here
  }

  Pickup *create_object() { return new Pickup(color); }

  using Base::parse;

protected:
  // Data to be parsed which will be used to construct the class, for example:

private:
  // Parse functions

  // Raw parse table
};

rtt_parser::Class_Parser<Pickup>::Class_Parser()
    : Class_Parser_Base<Pickup>(*this) {
  static bool first_time = true;
  if (first_time) {
    Base::add(Class_Parser_Base<Car>::table());
    first_time = false;
  }
}

template <>
shared_ptr<Pickup> rtt_parser::parse_class<Pickup>(Token_Stream &tokens) {
  rtt_parser::Class_Parser<Pickup> parser;
  return parser.parse(tokens);
}

//---------------------------------------------------------------------------//
// Ambiguous parse table.

class Ambiguous {};

template <>
class rtt_parser::Class_Parser<Ambiguous>
    : public Class_Parser_Base<Ambiguous> {
public:
  typedef Class_Parser_Base<Ambiguous> Base;

  Class_Parser();

  void check_completeness(Token_Stream &) {
    // No requirement here
  }

  Ambiguous *create_object() { return new Ambiguous(); }

  using Base::parse;

protected:
  // Data to be parsed which will be used to construct the class, for example:

private:
  // Parse functions

  void parse_moniker(Token_Stream &, int) {}

  // Raw parse table

  static Class_Parser_Keyword<Ambiguous> const raw_table[];
};

Class_Parser_Keyword<Ambiguous> const Class_Parser<Ambiguous>::raw_table[] = {
    {"moniker", &rtt_parser::Class_Parser<Ambiguous>::parse_moniker, 1, "main"},
    {"moniker", &rtt_parser::Class_Parser<Ambiguous>::parse_moniker, 0, "main"},
};

rtt_parser::Class_Parser<Ambiguous>::Class_Parser()
    : Class_Parser_Base<Ambiguous>(*this, raw_table,
                                   sizeof(raw_table) / sizeof(raw_table[0])) {}

template <>
shared_ptr<Ambiguous> rtt_parser::parse_class<Ambiguous>(Token_Stream &tokens) {
  rtt_parser::Class_Parser<Ambiguous> parser;
  return parser.parse(tokens);
}

//---------------------------------------------------------------------------//
// Once parse table.

struct Once {
  unsigned calls = 0;
};

template <>
class rtt_parser::Class_Parser<Once>
    : public Class_Parser_Base<Once, true, false> {
public:
  typedef Class_Parser_Base<Once, true, false> Base;

  Class_Parser();

  void check_completeness(Token_Stream &) {
    // No requirement here
  }

  Once *create_object() { return new Once{calls}; }

  using Base::parse;

  unsigned calls;

private:
  // Parse functions

  void parse_once(Token_Stream &, int) { calls++; }

  // Raw parse table

  static Class_Parser_Keyword<Once> const raw_table[];
};

Class_Parser_Keyword<Once> const Class_Parser<Once>::raw_table[] = {
    {"once", &rtt_parser::Class_Parser<Once>::parse_once, 1, "main"},
};

rtt_parser::Class_Parser<Once>::Class_Parser()
    : Class_Parser_Base<Once, true, false>(
          *this, raw_table, sizeof(raw_table) / sizeof(raw_table[0])),
      calls(0) {}

template <>
shared_ptr<Once> rtt_parser::parse_class<Once>(Token_Stream &tokens) {
  rtt_parser::Class_Parser<Once> parser;
  return parser.parse(tokens);
}

//--------------------------------------------- -----------------------------//
// Specialized token streams for extending code coverage.

class Error_Token_Stream : public Token_Stream {
public:
  void rewind() {}

  void comment(string const & /*err*/) {
    cout << "comment reported to Error_Token_Stream" << endl;
  }

protected:
  void report(Token const &, string const & /*err*/) {
    cout << "error reported to Error_Token_Stream" << endl;
  }

  void report(string const & /*err*/) {
    cout << "error reported to Error_Token_Stream" << endl;
  }

  Token fill_() { return Token(rtt_parser::ERROR, "error"); }
};

class Colon_Token_Stream : public Token_Stream {
public:
  Colon_Token_Stream() : count_(0) {}

  void rewind() {}

  void comment(string const & /*err*/) {
    cout << "comment reported to Colon_Token_Stream" << endl;
  }

protected:
  void report(Token const &, string const & /*err*/) {
    cout << "error reported to Colon_Token_Stream" << endl;
  }

  void report(string const & /*err*/) {
    cout << "error reported to Colon_Token_Stream" << endl;
  }

  Token fill_() {
    switch (count_++) {
    case 0:
      return Token(';', "");
    case 1:
      return Token(END, "end");
    case 2:
      return Token(EXIT, "");
    default:
      Insist(false, "bad case");
      return Token(rtt_parser::ERROR, ""); // dummy return to eliminate warning
    }
  }

private:
  unsigned count_;
};

//----------------------------------------------------------------------------//
void tstParse_Table(UnitTest &ut) {

  // Test keyword checks
  {
    // Invalid for lack of moniker
    Class_Parser_Keyword<Vehicle> k = {nullptr, nullptr, 0, "", ""};
    ut.check(!Is_Well_Formed_Keyword(k), "lack of moniker caught");

    // Invalid for lack of parse function
    k = {"moniker", nullptr, 0, "", ""};
    ut.check(!Is_Well_Formed_Keyword(k), "lack of func caught");

    // Invalid moniker
    k = {"0", &rtt_parser::Class_Parser<Vehicle>::parse_color, 0, "", ""};
    ut.check(!Is_Well_Formed_Keyword(k), "lack of moniker caught");

    // Invalid moniker
    k = {"and &", &rtt_parser::Class_Parser<Vehicle>::parse_color, 0, "", ""};
    ut.check(!Is_Well_Formed_Keyword(k), "lack of moniker caught");

    // Invalid moniker
    k = {"and&", &rtt_parser::Class_Parser<Vehicle>::parse_color, 0, "", ""};
    ut.check(!Is_Well_Formed_Keyword(k), "lack of moniker caught");

    // Valid
    k = {"_color", &rtt_parser::Class_Parser<Vehicle>::parse_color, 0, "", ""};
    ut.check(Is_Well_Formed_Keyword(k), "lack of moniker caught");
  }

  // Test parsing of a model class.
  {
    String_Token_Stream token_stream("BLUE\nend\n");

    shared_ptr<Vehicle> spectrum = parse_class<Vehicle>(token_stream);
    ut.check(token_stream.error_count() == 0, "Vehicle: no parse errors", true);
    ut.check(spectrum->color() == BLUE, "parsed BLUE");

    token_stream = String_Token_Stream("BLUISH GREEN\nend\n");

    spectrum = parse_class<Vehicle>(token_stream);
    ut.check(token_stream.error_count() == 0, "Vehicle: no parse errors", true);

    ut.check(spectrum->color() == BLUE_GREEN, "parsed BLUISH GREEN");

    token_stream = String_Token_Stream("color = BLACK\nend\n");

    spectrum = parse_class<Vehicle>(token_stream);
    ut.check(token_stream.error_count() == 0, "Vehicle: no parse errors", true);

    ut.check(spectrum->color() == BLACK, "parsed color = BLACK");

    // Deliberate error; unrecognized keyword

    token_stream = String_Token_Stream("COLOR = BLACK\nend\n");

    spectrum = parse_class<Vehicle>(token_stream);
    ut.check(token_stream.error_count() != 0 && spectrum == nullptr,
             "Vehicle: deliberate parse error");

    // Deliberate error; no end

    token_stream = String_Token_Stream("COLOR = BLACK\n");

    try {
      spectrum = parse_class<Vehicle>(token_stream);
      ut.failure("catches missing end");
    } catch (...) {
      ut.passes("catches missing end");
    }

    // Deliberate error; syntax after keyword

    token_stream = String_Token_Stream("COLOR = 1\n");

    try {
      spectrum = parse_class<Vehicle>(token_stream);
      ut.failure("catches internal syntax");
    } catch (...) {
      ut.passes("catches internal syntax");
    }

    // Deliberate error; non keyword

    token_stream = String_Token_Stream("1\n");

    try {
      spectrum = parse_class<Vehicle>(token_stream);
      ut.failure("catches internal syntax");
    } catch (...) {
      ut.passes("catches internal syntax");
    }

    // Deliberate error; I/O error

    Error_Token_Stream error_stream;

    try {
      spectrum = parse_class<Vehicle>(error_stream);
      ut.failure("catches i/o error");
    } catch (...) {
      ut.passes("catches i/o error");
    }
  }

  // Test parsing for case of empty class (supported as placeholder)
  // Also tests allow_exit and colon.
  {
    String_Token_Stream token_stream("");

    shared_ptr<Empty> empty = parse_class<Empty>(token_stream);

    ut.check(token_stream.error_count() == 0, "Empty: no parse errors");

    // Colon

    Colon_Token_Stream colon_stream;

    empty = parse_class<Empty>(colon_stream);
    ut.check(colon_stream.error_count() == 0, "colon treated as empty", true);
  }

  // Test inheritance
  {
    String_Token_Stream token_stream("BLUE\nend\n");

    shared_ptr<Pickup> spectrum = parse_class<Pickup>(token_stream);

    ut.check(spectrum->color() == BLUE, "parsed BLUE");
    ut.check(token_stream.error_count() == 0, "Pickup: no parse errors");
  }

  // Test detection of ambiguous keyword
  {
    String_Token_Stream token_stream("moniker\nend\n");

    try {
      shared_ptr<Ambiguous> spectrum = parse_class<Ambiguous>(token_stream);
      ut.failure("catch of ambiguous keyword");
    } catch (std::invalid_argument &) {
      ut.passes("catch of ambiguous keyword");
    }
  }

  // Test once parser
  {
    String_Token_Stream token_stream("once\nonce\nend\n");

    shared_ptr<Once> spectrum = parse_class<Once>(token_stream);
    ut.check(token_stream.error_count() == 0, "Vehicle: no parse errors", true);
    ut.check(spectrum->calls == 1, "called once");
  }

  return;
}

//---------------------------------------------------------------------------------------//

int main(int argc, char *argv[]) {
  ScalarUnitTest ut(argc, argv, release);
  try {
    tstParse_Table(ut);
  }
  UT_EPILOG(ut);
}

//---------------------------------------------------------------------------------------//
// end of tstParse_Table.cc
//---------------------------------------------------------------------------------------//
