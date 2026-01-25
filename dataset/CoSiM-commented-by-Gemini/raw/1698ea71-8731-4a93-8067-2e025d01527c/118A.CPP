/*
  Contest 118
  Problem A : String task
  Author : Rakesh Kumar
  Date: 17/01/2020
*/

#include <bits/stdc++.h>

inline bool vowel(char c) {
  if (c == 'a' || c == 'A' || c == 'o' || c == 'O' || c == 'y' || c == 'Y' ||
      c == 'e' || c == 'E' || c == 'u' || c == 'U' || c == 'i' || c == 'I')
    return true;
  return false;
}

int main() {
  std::string str;
  std::getline(std::cin, str);

  std::string result;
  for (char c : str) {
    if (!vowel(c)) {
      c = std::tolower(c);
      result += ".";
      result.push_back(c);
    }
  }
  std::cout << result << std::endl;

  return 0;
}
