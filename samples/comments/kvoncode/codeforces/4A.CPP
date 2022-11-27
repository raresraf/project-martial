#include <iostream>

int main() {
  int w = 0;
  std::string answer = "YES";
  std::cin >> w;

  if (w == 2 || w % 2 == 1) {
    answer = "NO";
  }

  std::cout << answer << std::endl;

  return 0;
}
