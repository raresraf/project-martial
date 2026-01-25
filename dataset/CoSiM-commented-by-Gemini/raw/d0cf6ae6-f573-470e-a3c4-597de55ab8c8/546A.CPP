/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/546/A
 *
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  short k, w;
  int n;

  cin >> k >> n >> w;

  for (int i = 1; i <= w; i++) {
    n -= i * k;
  }

  if (n > 0)
    n = 0;
  cout << abs(n) << '\n';
}
