#include <iostream>

int main() {
  std::string s;
  std::cin >> s;
  char c = s[0], count = 1;
  for (unsigned int i = 0; i < s.size(); ++i) {
    if (s[i] != c) {
      c = s[i];
      count = 1;
    } else if (s[i] == c) {
      ++count;
    }
    if (count == 7) {
      std::cout << "YES\n";
      return 0;
    }
  }
  std::cout << "NO\n";
}
