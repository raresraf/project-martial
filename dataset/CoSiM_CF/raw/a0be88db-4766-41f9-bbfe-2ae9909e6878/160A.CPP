#include <algorithm>
#include <iostream>
#include <vector>

struct greater {
  template <class T> bool operator()(T const &a, T const &b) const {
    return a > b;
  }
};

int main() {
  std::ios::sync_with_stdio(false);
  int i, n;
  std::cin >> n;

  std::vector<int> coins;
  int coin, max = 0;
  for (i = 0; i < n; ++i) {
    std::cin >> coin;
    coins.push_back(coin);
    max += coin;
  }

  float avg = float(max) / 2;
  std::sort(coins.begin(), coins.end(), greater());

  int taken_value = 0;
  for (i = 0; i < n; i++) {
    taken_value += coins[i];
    if (taken_value > avg)
      break;
  }

  std::cout << i + 1 << std::endl;
  return 0;
}