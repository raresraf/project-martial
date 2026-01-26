#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, a, b, c, d, e = 0;
  cin >> t;
  while (t--) {
    cin >> a;
    cin >> b;
    cin >> c;
    d = 0;
    if (a == 1) {
      d = d + 1;
    }
    if (b == 1) {
      d = d + 1;
    }
    if (c == 1) {
      d = d + 1;
    }
    if (d >= 2) {
      e = e + 1;
    }
  }
  cout << e;
}