#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int num_coins, total_sum{0};
  cin >> num_coins;
  vector<int> coins(num_coins);
  for (size_t i{0}; i < num_coins; ++i) {
    cin >> coins[i];
    total_sum += coins[i];
  }
  sort(begin(coins), end(coins));
  int min_coins_required{0}, sum{0};
  for (int i = num_coins - 1; i >= 0; --i) {
    sum += coins[i];
    ++min_coins_required;
    if (sum > total_sum - sum) {
      break;
    }
  }
  cout << min_coins_required << endl;
  return 0;
}
