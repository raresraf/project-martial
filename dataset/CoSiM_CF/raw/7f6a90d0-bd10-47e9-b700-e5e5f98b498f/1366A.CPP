//#include <bits/stdc++.h>
#include <iostream>
//#include <algorithm>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int t, a, b;
  cin >> t;

  while (t--) {
    cin >> a >> b;

    int minv = min(a, b);
    int maxv = max(a, b);

    if (minv <= (maxv >> 1)) {
      cout << minv;
    } else {

      int r1 = minv / 3;
      int r2 = maxv / 3;

      int r = r1 + r2;

      minv %= 3;
      maxv %= 3;

      if (minv + maxv >= 3) {
        r++;
      }

      cout << r;
    }

    cout << endl;
  }

  return 0;
}
