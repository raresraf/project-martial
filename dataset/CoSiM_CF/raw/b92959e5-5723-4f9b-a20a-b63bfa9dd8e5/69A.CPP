#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  int x = 0;
  int y = 0;
  int z = 0;
  while (t--) {
    int nX, nY, nZ;
    cin >> nX >> nY >> nZ;
    x += nX;
    y += nY;
    z += nZ;
  }
  if (x != 0 || y != 0 || z != 0) {
    cout << "NO";
  } else {
    cout << "YES";
  }
  return 0;
}
