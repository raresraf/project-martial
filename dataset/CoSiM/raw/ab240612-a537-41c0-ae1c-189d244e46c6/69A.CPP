#include <cmath>
#include <iostream>
#include <string>

using namespace std;

void solve() {
  int n;
  cin >> n;
  int a, b, c, x, y, z;
  x = y = z = 0;
  while (cin >> a >> b >> c) {
    x += a;
    y += b;
    z += c;
  }

  if (x || y || z)
    cout << "NO\n";
  else
    cout << "YES\n";
}

int main() {
  solve();
  return 0;
}