#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  int n, coins[100], sum = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> coins[i];
    sum += coins[i];
  }

  sort(coins, coins + n);
  int current_value = 0, ans = 0;
  for (int i = n - 1; i >= 0; i--) {
    if (current_value > sum) {
      cout << ans << endl;
      return 0;
    }
    current_value += coins[i];
    sum -= coins[i];
    ans++;
  }
  cout << ans << endl;
  return 0;
}

// g++ -std=c++11 -O2 -Wall 160A.cpp -o out