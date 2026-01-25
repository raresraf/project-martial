// Sagindyk Urazayev
// http://codeforces.com/problemset/problem/69/A

#include <iostream>

using namespace std;

int main() {
  int zres, yres, xres, n, i, x, y, z;
  xres = yres = zres = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z;
    xres = xres + x;
    yres = yres + y;
    zres = zres + z;
  }
  if ((xres == 0) && (yres == 0) && (zres == 0)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
