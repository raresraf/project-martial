#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  int n, k;
  std::string word;

  std::cin >> n;
  // std::cout << n;

  for (int i = 0; i < n; i++) {
    std::cin >> word;

    int wordLength = word.length();

    if (wordLength > 10) {
      word =
          word.at(0) + std::to_string(wordLength - 2) + word.at(wordLength - 1);
    }

    std::cout << word << std::endl;
  }

  return 0;
}
