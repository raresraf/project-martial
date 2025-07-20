// link : https://codeforces.com/contest/69/problem/A

#include <bits/stdc++.h>

using namespace std;

int n, a, b, c, ans1, ans2, ans3;

main() {

  cin >> n;

  for (int i = 0; i < n; i++) {

    cin >> a >> b >> c;

    ans1 += a;
    ans2 += b;
    ans3 += c;
  }

  if ((!ans1) && (!ans2) && (!ans3))
    cout << "YES" << endl;

  else
    cout << "NO" << endl;
}
