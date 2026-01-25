#include <iostream>
using namespace std;

int main() {
  int n, x, y, z, a, b, c;
  a = b = c = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    a += x;
    b += y;
    c += z;
  }
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}