#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
  int t;
  cin >> t;
  while (t--) {
    int n, m;
    cin >> n >> m;
    bool flag = false;
    for (int p = 0; p < n; p++) {
      int mat[2][2];
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          cin >> mat[i][j];
        }
      }
      if (mat[0][1] == mat[1][0])
        flag = true;
    }
    if (flag && (m * m) % 4 == 0)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}