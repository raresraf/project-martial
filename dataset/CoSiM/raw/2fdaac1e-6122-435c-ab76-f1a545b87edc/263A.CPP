#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  int number, n = 5;
  int mid_point = (n + 1) / 2; // only for odd n
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      std::cin >> number;
      if (number == 1) {
        std::cout << abs(i - mid_point) + abs(j - mid_point) << std::endl;
      }
    }
  }
  return 0;
}