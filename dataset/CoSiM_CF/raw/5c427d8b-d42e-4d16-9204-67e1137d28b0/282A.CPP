/*
  Contest 282
  Problem A : Bit++
  Author : Rakesh Kumar
  Date: 17/01/2020
*/

#include <bits/stdc++.h>

int main() {
  int n = 0;
  std::cin >> n;
  std::cin.ignore();

  int x = 0;
  while (n--) {
    std::string str;
    std::getline(std::cin, str);
    if (str[0] == '+' || str[str.size() - 1] == '+')
      ++x;
    else
      --x;
  }

  std::cout << x << std::endl;

  return 0;
}
