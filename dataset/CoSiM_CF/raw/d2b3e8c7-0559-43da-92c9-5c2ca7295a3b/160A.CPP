// Codeforces
// 160A - Twins

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

int main() {
  int n;
  cin >> n;

  vector<int> coins;
  int c;
  for (int i = 0; i < n; i++) {
    cin >> c;
    coins.push_back(c);
  }

  int sum = accumulate(coins.begin(), coins.end(), 0);
  sort(coins.begin(), coins.end(), greater<int>());

  int i = 0, total = 0, nbrCoins = 0;

  while (total <= sum / 2.0) {
    total += coins[i];
    i++;
    nbrCoins++;
  }

  cout << nbrCoins;

  return 0;
}
