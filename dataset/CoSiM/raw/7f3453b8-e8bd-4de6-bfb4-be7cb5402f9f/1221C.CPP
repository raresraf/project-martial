#include <bits/stdc++.h>
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
using namespace std;

int t, c, m, n;

int main() {
  cin >> t;
  while (t--) {
    cin >> c >> m >> n;
    cout << min(min(c, m), (c + m + n) / 3) << '\n';
  }

  return 0;
}
