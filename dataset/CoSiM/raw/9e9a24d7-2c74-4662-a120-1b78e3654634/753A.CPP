#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, i;
  cin >> n;
  int dp[n] = {};
  i = 1;

  while (i <= n) {
    dp[i] = i;
    n -= i;
    i++;
  }
  if (n) {
    dp[i - 1] += n;
  }
  k = i;
  cout << k - 1 << endl;
  for (i = 1; i < k; i++)
    cout << dp[i] << " ";
}
