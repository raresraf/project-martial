#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a, b, x, y, z, max = 0;
  cin >> n;
  cin >> a >> b;
  x = a + b;
  n = n - 1;
  while (n--) {
    cin >> a >> b;
    y = x - a;
    z = y + b;
    if (z > max)
      max = z;
    x = z;
  }
  cout << max << endl;
}
