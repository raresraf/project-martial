#include <iostream>
#include <stdio.h>
using namespace ::std;
main() {
  ios_base::sync_with_stdio(false); // Fast I/O
  int k, n, w;
  cin >> k >> n >> w;
  long int t = k * (w * (w + 1)) / 2;
  if (n >= t)
    cout << 0;
  else
    cout << t - n;
  return 0;
}