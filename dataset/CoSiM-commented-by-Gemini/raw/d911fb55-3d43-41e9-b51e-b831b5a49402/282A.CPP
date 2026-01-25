#include <iostream>
#include <regex>
#include <string>

int main() {
  int n;
  int counter = 0;

  std::cin >> n;

  for (int i = 0; i < n; i++) {
    std::string command;
    std::cin >> command;

    if (std::regex_match(command, std::regex("(.*)(--)(.*)")))
      counter--;
    else
      counter++;
  }

  std::cout << counter << std::endl;

  return 0;
}
