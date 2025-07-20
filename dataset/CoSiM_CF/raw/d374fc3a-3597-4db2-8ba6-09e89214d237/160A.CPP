#include <bits/stdc++.h>

using namespace std;

int solve(int sum, vector<int> coins) {
  sort(coins.begin(), coins.end(), greater<int>());
  int cnt = 0, run_sum = 0;
  for (int coin : coins) {
    run_sum += coin;
    cnt++;
    if (run_sum > sum / 2) {
      break;
    }
  }

  return cnt;
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  cin >> n;
  vector<int> coins;
  int coin;
  int sum = 0;
  while (n--) {
    cin >> coin;
    sum += coin;
    coins.push_back(coin);
  }

  cout << solve(sum, coins) << endl;
}
