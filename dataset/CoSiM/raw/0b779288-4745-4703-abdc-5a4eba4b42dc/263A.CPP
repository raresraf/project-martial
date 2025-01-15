// CF https://codeforces.com/contest/263/problem/A

#include <bits/stdc++.h>

using namespace std;

int32_t main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int n = 5, m = 5;
  int mat[n][m];
  int i = 0, j = 0;
  for (i = 0; i < n; i++) {
    int flag = 0;
    for (j = 0; j < m; j++) {
      int x;
      cin >> x;
      if (x == 1) {
        flag = 1;
        break;
      }
    }
    if (flag == 1)
      break;
  }
  cout << abs(2 - i) + abs(2 - j) << "\n";
  return 0;
}
