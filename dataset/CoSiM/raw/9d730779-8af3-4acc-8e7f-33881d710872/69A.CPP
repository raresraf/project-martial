#include <iostream>

using namespace std;

int main() {
  int n, x, y, z, x0 = 0, y0 = 0, z0 = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    x0 = x0 + x;
    y0 = y0 + y;
    z0 = z0 + z;
  }
  if (x0 == 0 && y0 == 0 && z0 == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
