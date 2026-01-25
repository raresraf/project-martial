#include <bits/stdc++.h>
using namespace std;
int main() {
  int n = 0, x = 0, y = 0, z = 0;
  cin >> n;
  while (n--) {
    int xa, ya, za;
    cin >> xa >> ya >> za;
    x += xa;
    y += ya;
    z += za;
  }
  if (x == 0 and y == 0 and z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}