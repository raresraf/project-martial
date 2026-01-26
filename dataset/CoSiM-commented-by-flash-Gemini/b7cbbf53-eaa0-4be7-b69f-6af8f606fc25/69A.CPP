#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, a, b, c, ca = 0, cb = 0, cc = 0;
  cin >> t;
  while (t--) {
    cin >> a >> b >> c;
    ca += a;
    cb += b;
    cc += c;
  }
  if (ca == 0 && cb == 0 && cc == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
