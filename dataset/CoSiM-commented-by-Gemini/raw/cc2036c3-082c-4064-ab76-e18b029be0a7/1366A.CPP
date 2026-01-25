#include <bits/stdc++.h>
using namespace std;
int t, a, b, res;
main() {
  cin >> t;
  while (t--) {
    res = 0;
    cin >> a >> b;
    cout << min(min(a, b), (a + b) / 3) << '\n';
  }
}
