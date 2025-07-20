#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, x = 0, y = 0, z = 0, x1, y1, z1;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> x1 >> y1 >> z1;
    x += x1;
    y += y1;
    z += z1;
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}