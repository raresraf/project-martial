// https://codeforces.com/problemset/problem/69/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int tc;
  cin >> tc;
  int x, y, z;
  x = y = z = 0;
  int a, b, c;
  while (tc--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  string res = (x == 0 && y == 0 && z == 0) ? "YES" : "NO";

  cout << res << endl;
}
