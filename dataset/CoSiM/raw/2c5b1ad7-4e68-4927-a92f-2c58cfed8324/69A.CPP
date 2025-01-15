#include <iostream>
using namespace std;
int main() {
  int n, fx = 0, fy = 0, fz = 0, x, y, z;
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    fx += x;
    fy += y;
    fz += z;
  }
  if (fx == 0 && fy == 0 && fz == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
