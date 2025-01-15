#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i;
  cin >> n;

  int dp[n + 1] = {};
  dp[3] = 6;
  for (i = 4; i <= n; i++) {
    dp[i] = dp[i - 1] + i * (i - 1);
  }
  cout << dp[n];
}
