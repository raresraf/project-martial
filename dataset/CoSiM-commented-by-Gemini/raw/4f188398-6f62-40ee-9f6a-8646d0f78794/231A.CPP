#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, n, cnt = 0, re = 0, p, v, t;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> p >> v >> t;
    if (p == 1) {
      cnt++;
    }
    if (v == 1) {
      cnt++;
    }
    if (t == 1) {
      cnt++;
    }
    if (cnt >= 2) {
      re++;
    }
    cnt = 0;
  }
  cout << re;
  return 0;
}
