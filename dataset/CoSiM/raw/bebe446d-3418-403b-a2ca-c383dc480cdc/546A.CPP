#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, b, i, j, l, o = 0, k, w;
  cin >> k;
  cin >> n;
  cin >> w;
  b = (k * (w + 1) * w / 2) - n;
  if (b < 0) {
    b = 0;
  }
  cout << b;
  return 0;
}
