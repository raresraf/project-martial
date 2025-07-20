#include <cctype>
#include <iostream>
#include <string>

int main() {
  std::string slowo;
  std::cin >> slowo;
  if (slowo[0] > 90) {
    slowo[0] -= 32;
  }
  std::cout << slowo;

  return 0;
}