#include <iostream> // std::cout
#include <locale>   // std::locale, std::tolower
#include <string>   // std::string

int compare(char a, char b) {
  std::locale loc;
  char lowerA = std::tolower(a, loc);
  char lowerB = std::tolower(b, loc);

  if (lowerA < lowerB)
    return -1;
  else if (lowerA > lowerB)
    return 1;
  else
    return 0;
}

int main() {
  std::string firstString, secondString;
  int stringLength;
  int answer = 0;

  std::cin >> firstString >> secondString;

  stringLength = firstString.length();

  for (int i = 0; i < stringLength; i++) {
    int temp = compare(firstString[i], secondString[i]);

    if (temp == -1) {
      answer = -1;
      break;
    } else if (temp == 1) {
      answer = 1;
      break;
    }
  }

  std::cout << answer << std::endl;

  return 0;
}
