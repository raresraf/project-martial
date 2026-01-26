
#include <bits/stdc++.h>
using namespace std;
int main() {
  int a, b, c, t;
  int x = 0, y = 0, z = 0;
  cin >> t;
  while (t--) {

    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
