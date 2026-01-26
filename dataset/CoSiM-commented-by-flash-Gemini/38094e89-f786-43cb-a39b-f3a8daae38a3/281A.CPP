#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <string>

int main() {
  std::string input;
  std::cin >> input;

  input[0] = toupper(input[0]);

  std::cout << input << std::endl;

  return 0;
}
