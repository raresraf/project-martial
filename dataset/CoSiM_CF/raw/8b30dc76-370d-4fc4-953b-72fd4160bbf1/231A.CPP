#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x, y, z, ok = 0, yes = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    if (x == 1)
      ok++;
    if (y == 1)
      ok++;
    if (z == 1)
      ok++;
    if (ok >= 2) {
      yes++;
    }
    ok = 0;
  }
  cout << yes << endl;
}
