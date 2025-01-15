#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, coin_value;
  vector<int> coins;

  cin >> n;
  while (n--) {
    cin >> coin_value;
    coins.push_back(coin_value);
  }
  sort(coins.begin(), coins.end());

  int total_sum = 0;
  for (int i = 0; i < coins.size(); i++) {
    total_sum += coins[i];
  }
  int gt_sum = 0;
  int min_coins = 0;
  for (int i = coins.size() - 1; i >= 0; i--) {
    gt_sum += coins[i];
    min_coins++;
    if (gt_sum > total_sum - gt_sum) {
      break;
    }
  }

  cout << min_coins << endl;

  return 0;
}
