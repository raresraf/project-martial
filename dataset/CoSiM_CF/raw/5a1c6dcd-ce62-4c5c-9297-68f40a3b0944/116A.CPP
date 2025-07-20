// link : https://codeforces.com/problemset/problem/116/A

#include <bits/stdc++.h>

using namespace std;

main() {

  int n, ans1 = 0, k = 0, r = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {

    int g1, g2;

    cin >> g1 >> g2;

    if (i == 0)
      r = g1;

    if (i != 0)
      k -= g1;

    k += g2;

    ans1 = max(ans1, k);
  }

  cout << ans1;
}
