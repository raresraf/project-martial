#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int a, b, c;
  int x, y, z;
  x = y = z = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }
  if ((x == 0) && (y == 0) && (z == 0))
    cout << "YES\n";
  else
    cout << "NO\n";
}
