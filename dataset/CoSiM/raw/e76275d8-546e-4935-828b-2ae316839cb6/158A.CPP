/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/158/A
 *
 */

#include <bits/stdc++.h>
#define PB push_back

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  short n, k, a, c{0};
  vector<short> p;
  cin >> n >> k;

  while (cin >> a) {
    p.PB(a);
  }

  a = p[k - 1];
  for (short s : p)
    s >= a &&s != 0 ? c++ : 0;

  cout << c << '\n';
}
