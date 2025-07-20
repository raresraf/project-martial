#include <iostream>

int main() {
  std::string word;
  int n;
  int len = 0;
  std::cin >> n;
  while (n--) {
    std::cin >> word;
    len = word.length();
    if (len <= 10) {
      std::cout << word << std::endl;
    } else {
      std::cout << word[0] << len - 2 << word[len - 1] << std::endl;
    }
  }
  return 0;
}