#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, a, b, m = 0, n = -1;
  cin >> t;
  while (t > 0) {
    t--;
    cin >> a >> b;
    m = m - a + b;
    n = max(m, n);
  }
  cout << n;
}
