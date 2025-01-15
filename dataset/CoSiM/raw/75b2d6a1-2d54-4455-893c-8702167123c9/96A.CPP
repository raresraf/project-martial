#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  std::string standing;
  std::cin >> standing;

  if (standing.find("0000000") != std::string::npos ||
      standing.find("1111111") != std::string::npos) {
    std::cout << "YES" << std::endl;
  } else {
    std::cout << "NO" << std::endl;
  }
  return 0;
}