#include <bits/stdc++.h>
using namespace std;

int x, y, z;
int main() {
  int n, a, b, c;
  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }
  if (x == 0 & y == 0 & z == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}