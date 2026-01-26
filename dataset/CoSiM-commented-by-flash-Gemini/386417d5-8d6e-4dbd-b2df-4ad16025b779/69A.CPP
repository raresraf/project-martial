#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x = 0, y = 0, z = 0;

  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;

    x += a;
    y += b;
    z += c;
  }

  (0 == x && x == y && y == z) ? cout << "YES" : cout << "NO";
  cout << endl;

  return 0;
}
