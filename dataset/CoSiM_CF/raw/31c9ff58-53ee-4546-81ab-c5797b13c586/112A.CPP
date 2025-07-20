#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, b;
  int l = 0, r = 0, e, i, f, h = 0;
  cin >> a;
  cin >> b;
  for (i = 0; i < a.size(); i++) {
    e = a[i];
    if (e < 91) {
      e = e + 32;
    }
    l = e;
    f = b[i];
    if (f < 91) {
      f = f + 32;
    }
    r = f;
    if (l > r) {
      cout << 1;
      h = h + 1;
      break;
    } else if (l < r) {
      cout << -1;
      h = h + 1;
      break;
    }
  }
  if (h == 0) {
    cout << 0;
  }
}