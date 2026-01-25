#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int n, x = 0, y = 0, z = 0, a, b, c;
  cin >> n;

  while (n--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  (x == 0 && y == 0 && z == 0) ? cout << "YES\n" : cout << "NO\n";
}