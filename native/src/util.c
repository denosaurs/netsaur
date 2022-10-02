#include <include/util.h>

unsigned char seeded = 0;
unsigned char gauss_return = 0;
float gauss_value = 0.0;

float gauss_rand() {
  if (seeded == 0) {
    srand(time(NULL));
    seeded = 1;
  }

  if (gauss_return == 1) {
    gauss_return = 0;
    return gauss_value;
  }

  float u, v, r;
  u = ((float)rand() * 2.0) / RAND_MAX - 1.0;
  v = ((float)rand() * 2.0) / RAND_MAX - 1.0;
  r = u * u + v * v;
  if (r == 0.0 || r > 1.0) return gauss_rand();
  float c = sqrt(-2.0 * log(r) / r);
  gauss_value = v * c;
  gauss_return = 1;
  return u * c;
}
