#include <iostream>
using namespace std;

int main() {
  int n, x, y, z;
  cin >> n;
  int i, x1 = 0, y1 = 0, z1 = 0;
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z;
    x1 += x;
    y1 += y;
    z1 += z;
  }
  if (x1 == 0 && y1 == 0 && z1 == 0)
    cout << "YES" << endl;
  else
    cout << "NO";
  return 0;
}
