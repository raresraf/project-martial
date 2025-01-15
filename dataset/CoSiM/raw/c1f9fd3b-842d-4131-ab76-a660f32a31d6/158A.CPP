#include <cstdio>
#include <iostream>

using namespace std;

main() {
  int p[100];
  int n, i, k, x = 0, y;

  cin >> n >> k;

  for (i = 0; i < n; i++)
    cin >> p[i];
  for (i = 0; i < n; i++) {
    if ((p[i] >= p[k - 1]) && (p[i] > 0)) {
      x++;
    }
  }

  cout << x;
}