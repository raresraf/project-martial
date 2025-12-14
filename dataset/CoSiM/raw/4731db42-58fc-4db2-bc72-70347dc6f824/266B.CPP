// input
// 5 1
// BGGBG
// output
// GBGGB

#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n, t;
  cin >> n >> t;
  string a;
  cin >> a;

  while (t-- > 0) {
    for (int i = 1; i < n; i++) {
      if (a[i] == 'G' && a[i - 1] == 'B') {
        a[i] = 'B';
        a[i - 1] = 'G';
        ++i;
      }
    }
  }
  cout << a;
  return 0;
}
