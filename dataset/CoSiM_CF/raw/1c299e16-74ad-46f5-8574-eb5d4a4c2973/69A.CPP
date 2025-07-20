#include <iostream>
using namespace std;

int main() {
  int a, b, n, c, x = 0, y = 0, z = 0, res = 0;
  cin >> n;
  while (n != 0) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
    --n;
    cout << x << "  " << y << "  " << z << "\n";
  }

  if (x == 0 && y == 0 && z == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
