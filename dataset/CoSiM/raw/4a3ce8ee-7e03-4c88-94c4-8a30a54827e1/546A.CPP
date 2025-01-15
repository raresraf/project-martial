#include <iostream>

int main() {
  int k, n, w;
  std::cin >> k >> n >> w;
  int required = k * w * (w + 1) / 2;
  std::cout << ((required > n) ? required - n : 0) << '\n';
}
