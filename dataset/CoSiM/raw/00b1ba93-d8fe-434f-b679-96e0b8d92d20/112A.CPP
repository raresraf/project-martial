#include <iostream>
#include <string>

int main() {
  std::string s1, s2;

  std::cin >> s1 >> s2;

  bool done = false;

  for (int i = 0; i < s1.size(); i++) {
    char s1_char = tolower(s1.at(i));
    char s2_char = tolower(s2.at(i));

    if (s1_char > s2_char) {
      if (!done)
        std::cout << 1;
      { done = true; }
    }
    if (s2_char > s1_char) {
      if (!done)
        std::cout << -1;
      { done = true; }
    }
  }
  if (!done)
    std::cout << 0;
}