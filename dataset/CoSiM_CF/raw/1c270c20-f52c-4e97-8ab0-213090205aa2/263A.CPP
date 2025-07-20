#include <cmath>
#include <iostream>

int main() {
  int iMax = 5, jMax = 5;
  int current = 0;

  for (int i = 0; i < iMax; i++) {
    for (int j = 0; j < jMax; j++) {

      std::cin >> current;

      if (1 == current) {
        std::cout << std::abs(2 - i) + std::abs(2 - j) << std::endl;
      }
    }
  }

  return 0;
}
