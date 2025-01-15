#include <algorithm>
#include <iostream>
#include <regex>

int main() {
  std::regex vowel("[aeiouy]");
  std::regex consonant("[a-z&&[^aeiouy]]");
  std::string inputString;
  std::cin >> inputString;
  transform(inputString.begin(), inputString.end(), inputString.begin(),
            ::tolower);
  inputString = std::regex_replace(inputString, vowel, "");
  for (int i = 0; i < inputString.length(); i++) {
    std::cout << '.' << inputString[i];
  }
  return 0;
}