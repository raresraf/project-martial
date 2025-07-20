#include <cctype>
#include <iostream>
#include <string>

int main() {
  std::string slowo;
  std::cin >> slowo;
  int dlugosc = slowo.length();
  std::string output = "";
  char c;
  for (int i = 0; i < dlugosc; i++) {
    if (slowo[i] == 65 || slowo[i] == 79 || slowo[i] == 89 || slowo[i] == 69 ||
        slowo[i] == 85 || slowo[i] == 73) {
      continue;
    } else if (slowo[i] == 97 || slowo[i] == 111 || slowo[i] == 121 ||
               slowo[i] == 101 || slowo[i] == 117 || slowo[i] == 105) {
      continue;
    } else {
      output += ".";
      c = slowo[i];
      output += tolower(c);
    }
  }

  std::cout << output;

  return 0;
}