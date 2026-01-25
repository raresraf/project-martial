//
// Created by Conor on 5/28/2020.
//

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  vector<int> coins;

  cin >> n;

  int total = 0;
  int val;
  for (int i = 0; i < n; ++i) {
    cin >> val;
    coins.push_back(val);
    total += coins[i];
  }

  sort(coins.begin(), coins.end(), greater<int>());

  int coin_count = 0;
  int share = 0;

  for (int i : coins) {
    coin_count++;
    total -= i;
    share += i;
    if (share > total) {
      cout << coin_count;
      break;
    }
  }

  return 0;
}