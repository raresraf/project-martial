#include <iostream>

int main() {
  std::string number;
  long long luckyNumbers = 0;
  std::cin >> number;
  for (long long i = 0; i < number.length(); i++) {
    if (number[i] == '4' || number[i] == '7') {
      luckyNumbers++;
    }
  }
  if (luckyNumbers == 4 || luckyNumbers == 7 || luckyNumbers == 47 ||
      luckyNumbers == 74 || luckyNumbers == 447) {
    std::cout << "YES";
  } else if (luckyNumbers == 474 || luckyNumbers == 477 ||
             luckyNumbers == 744 || luckyNumbers == 747 ||
             luckyNumbers == 774 || luckyNumbers == 777) {
    std::cout << "YES";
  } else {
    std::cout << "NO";
  }

  return 0;
}