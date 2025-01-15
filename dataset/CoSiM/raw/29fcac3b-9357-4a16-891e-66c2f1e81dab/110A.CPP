#include <iostream>

bool checkLuckyDigit(int n) {
  // assume n is single digit
  if (n >= 10)
    return false;
  if ((n != 4) && (n != 7))
    return false;
  return true;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::string number;
  std::cin >> number;
  int tmp, count = 0;
  for (int i = 0; i < number.length(); ++i) {
    tmp = number[i] - '0';
    if (checkLuckyDigit(tmp))
      ++count;
  }
  if (checkLuckyDigit(count))
    std::cout << "YES" << std::endl;
  else
    std::cout << "NO" << std::endl;
  return 0;
}