#include <iostream>
#include <regex>
#include <string>

int main() {
  std::string input("");
  std::regex upCase("[A-Z]");
  std::regex downCase("[a-z]");
  std::regex vowels("[aoyeuiAOYEUI]");

  std::cin >> input;

  std::string withoutVowels = std::regex_replace(input, vowels, "");
  std::transform(withoutVowels.begin(), withoutVowels.end(),
                 withoutVowels.begin(), ::tolower);

  std::string withPoints = std::regex_replace(withoutVowels, downCase, ".$&");

  std::cout << withPoints << std::endl;
  return 0;
}
