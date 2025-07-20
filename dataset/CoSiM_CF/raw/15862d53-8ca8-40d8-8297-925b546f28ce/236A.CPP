#include <iostream>

#define ALPHABET_COUNT 26

int main() {
  char c;
  bool seen[ALPHABET_COUNT] = {0};
  int count = 0;
  while (std::cin >> c) {
    if (!seen[c - 'a']) {
      seen[c - 'a'] = true;
      ++count;
      if (count == 26) {
        std::cout << "CHAT WITH HER!\n";
        return 0;
      }
    }
  }
  std::cout << ((count % 2) ? "IGNORE HIM!" : "CHAT WITH HER!") << '\n';
}
