
/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/116/A
 *
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  short n, a, b, p(0), mp(0);
  cin >> n;
  for (short i = 0; i < n; i++) {
    cin >> a >> b;
    p -= a;
    p += b;
    if (p > mp)
      mp = p;
  }
  cout << mp << '\n';
}
