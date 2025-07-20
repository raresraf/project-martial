/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/96/A
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int c(1);
  string a;
  cin >> a;
  for (int i = 0; i < a.length(); i++) {
    c++;
    if (a[i] != a[i - 1]) {
      c = 1;
    } else if (c >= 7) {
      break;
    }
  }
  a = c >= 7 ? "YES" : "NO";
  cout << a;
}
