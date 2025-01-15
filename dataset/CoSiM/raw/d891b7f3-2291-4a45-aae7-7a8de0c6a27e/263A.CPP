#include <cmath>
#include <iostream>

#define SIZE 5

int takeInput() {
  int temp, ret;
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      std::cin >> temp;
      if (temp == 1)
        ret = 5 * i + j;
    }
  }
  return ret;
}

int main() {
  int position = takeInput();
  int column = position % 5, row = position / 5;
  int val = std::abs(column - 2) + std::abs(row - 2);
  std::cout << val << '\n';
}
