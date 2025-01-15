#include <iostream>
#include <string>

int main() {
  std::string s;
  do {
    std::cin >> s;
  } while (s.size() > 100 || s.size() < 1);
  if (s.size() < 2) {
    std::cout << s;
    return 0;
  }
  for (auto i = 0; i < s.size() - 2; i += 2) {
    if (s[i] > s[i + 2]) {
      std::swap(s[i], s[i + 2]);
      i = -2;
    }
  }
  std::cout << s;
  return 0;
}
