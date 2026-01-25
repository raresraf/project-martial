#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int m, k, i, w, f1, f2, p = 0, x;
  long int n;
  cin >> k >> n >> w;

  for (i = 1; i <= w; i++) {
    p = p + k * i;
  }
  x = p - n;
  if (x <= 0) {
    cout << 0;
  } else {
    cout << x;
  }
}
