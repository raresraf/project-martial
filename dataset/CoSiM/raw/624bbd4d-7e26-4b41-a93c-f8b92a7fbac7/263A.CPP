#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[5][5];
  int n, m;
  int ans;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] == 1) {
        ans = abs(i - 2) + abs(j - 2);
        // cout<<i<<j;
      }
    }
  }
  cout << ans;
}
