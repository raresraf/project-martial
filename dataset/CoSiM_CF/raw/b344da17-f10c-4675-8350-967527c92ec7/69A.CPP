#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x, y, z, tx = 0, ty = 0, tz = 0;
  while (n--) {
    cin >> x >> y >> z;
    tx += x, ty += y, tz += z;
  }
  if (tx == 0 && ty == 0 && tz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
