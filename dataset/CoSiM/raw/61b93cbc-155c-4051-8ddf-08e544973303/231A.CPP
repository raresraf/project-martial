#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int result = 0;
  int dp[n][3];
  for (int i = 0; i < n; i++) {
    int local_count = 0;
    for (int j = 0; j < 3; j++) {
      cin >> dp[i][j];
      if (dp[i][j] == 1) {
        local_count++;
      }
    }
    if (local_count >= 2)
      result++;
  }

  cout << result;

  return 0;
}