#include <iostream>

int main(int argc, char *argv[]) {
  int i = 0;
  int j = 0;

  char ch;
  while (std::cin >> ch) {
    if (ch == '1')
      break;

    ++i;
    if (i >= 5) {
      i = 0;
      ++j;
    }
  }

  std::cout << std::abs(2 - i) + std::abs(2 - j) << '\n';
}
