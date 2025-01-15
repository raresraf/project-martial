#include <iostream>
#include <stdlib.h> /* abs */
#include <string>
using namespace std;

int main() {
  long int k, n, w, tot = 0;
  cin >> k >> n >> w;

  for (int i = 1; i <= w; i++) {
    tot = tot + i * k;
  }
  if (n - tot < 0) {
    cout << abs(tot - n);
  } else {
    cout << 0;
  }
  return 0;
}
