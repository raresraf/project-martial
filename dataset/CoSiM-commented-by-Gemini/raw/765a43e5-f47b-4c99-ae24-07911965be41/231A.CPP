/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/231/A
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a, b, c, r;
  cin >> n;
  r = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if (a + b + c >= 2)
      ++r;
  }
  cout << r << endl;
  return 0;
}
