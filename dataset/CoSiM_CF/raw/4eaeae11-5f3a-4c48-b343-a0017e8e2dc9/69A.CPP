#include <bits/stdc++.h>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  long long int n, si = 0, sj = 0, sk = 0, i, j, k;
  cin >> n;
  while (n--) {
    cin >> i >> j >> k;
    si = si + i;
    sj += j;
    sk += k;
  }
  if (si == 0 && sj == 0 && sk == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}