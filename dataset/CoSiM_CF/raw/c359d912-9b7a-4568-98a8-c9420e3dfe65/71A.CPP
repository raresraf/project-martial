/*

Problem description:

https://codeforces.com/problemset/problem/71/A

*/

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, crlen;
  string cr;
  cin >> n;
  for (int i = 0; i <= n; i++) {
    getline(cin, cr);
    crlen = cr.length();
    if (crlen > 10) {
      cout << cr[0] << crlen - 2 << cr[crlen - 1] << endl;
      continue;
    }
    cout << cr << endl;
  }
  return 0;
}
