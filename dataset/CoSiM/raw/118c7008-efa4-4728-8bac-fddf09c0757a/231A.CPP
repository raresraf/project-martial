// link : https://codeforces.com/problemset/problem/231/A

#include <bits/stdc++.h>

using namespace std;

int t, k;

main() {

  cin >> t;

  while (t--) {

    int a[3];

    cin >> a[0] >> a[1] >> a[2];

    sort(a, a + 3);

    if (a[1] != 0 && a[2] != 0)
      k++;
  }

  cout << k;
}
