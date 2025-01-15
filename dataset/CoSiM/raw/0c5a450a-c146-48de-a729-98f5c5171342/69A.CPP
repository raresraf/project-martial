#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, x, y, z;
  int sx, sy, sz;
  sx = sy = sz = 0;
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    sx += x, sy += y, sz += z;
  }

  if (!sx && !sy && !sz) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}