#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  std::string word;
  std::cin >> word;

  word[0] = toupper(word[0]);

  std::cout << word << std::endl;

  return 0;
}