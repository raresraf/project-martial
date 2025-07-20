#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, i, ans(0), temp(0), tot(0);
  cin >> n;
  int coins[n];
  for (i = 0; i < n; i++) {
    cin >> coins[i];
    tot += coins[i];
  }
  sort(coins, coins + n);
  for (i = n - 1; i >= 0 && temp <= tot / 2; i--) {
    temp += coins[i];
    ans++;
    if (temp > tot / 2)
      break;
  }
  cout << ans << endl;
  return 0;
}
