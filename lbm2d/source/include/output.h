#ifndef OUTPUT_H
#define OUTPUT_H
#include <cstdlib>
#include <string>
#include <typedefs.h>
#include <params.h>

class Output {

public:

  Output();

  void write_view(const std::string &fn, ScalarField::HostMirror data, Params &params);

  ~Output();

  FILE *file;

  size_t frame = 0;

};

#endif // OUTPUT_H
