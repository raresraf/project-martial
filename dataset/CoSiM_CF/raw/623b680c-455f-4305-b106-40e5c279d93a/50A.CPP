#include <iostream>

int main() {
  int m, n;
  std::cin >> m >> n;
  int board = m * n;
  int output = 0;

  if (board % 2 == 0) {
    output = board / 2;
  } else {
    output = board / 2;
  }

  std::cout << output;

  return 0;
}