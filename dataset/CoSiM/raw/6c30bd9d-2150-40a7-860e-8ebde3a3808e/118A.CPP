#include <iostream>

#define EMPTY_STRING ""

bool isVowel(char c) {
  return ((c == 'a') || (c == 'A') || (c == 'e') || (c == 'E') || (c == 'i') ||
          (c == 'I') || (c == 'o') || (c == 'O') || (c == 'u') || (c == 'U') ||
          (c == 'y') || (c == 'Y'));
}

std::string replace(char c) {
  std::string ret = EMPTY_STRING;
  if (isVowel(c))
    return ret;
  ret += ".";
  if ((c >= 'a') && (c <= 'z'))
    ret += c;
  else if ((c >= 'A') && (c <= 'Z'))
    ret += c - 'A' + 'a';
  return ret;
}

int main() {
  std::string s;
  std::cin >> s;
  for (unsigned int i = 0; i < s.size(); ++i) {
    std::cout << replace(s[i]);
  }
  std::cout << '\n';
}
