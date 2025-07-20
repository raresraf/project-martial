#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  // your code goes here
  int n;
  cin >> n;
  int a[105];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  sort(a, a + n);
  int dp[104];
  dp[0] = a[0];
  for (int i = 1; i < n; i++) {
    dp[i] = dp[i - 1] + a[i];
  }
  int sum = 0;
  int ans = 0;
  for (int i = n - 1; i >= 0; i--) {
    ans++;
    sum += a[i];
    if (sum > dp[i - 1]) {
      break;
    }
  }
  cout << ans << endl;
  return 0;
}