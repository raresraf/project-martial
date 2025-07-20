#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  int x(0), y(0), z(0), xx, yy, zz;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> xx >> yy >> zz;
    x += xx;
    y += yy;
    z += zz;
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
