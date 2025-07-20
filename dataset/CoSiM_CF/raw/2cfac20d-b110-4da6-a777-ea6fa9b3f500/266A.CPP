/*
  Contest 266
  Problem A : Stones on the table
  Author : Rakesh Kumar
  Date: 21/01/2020
*/

#include <bits/stdc++.h>

int main() {
  int n = 0;
  std::cin >> n;

  int r = 0;
  char p = ' ';
  std::cin >> p;
  --n;

  while (n--) {
    char c = ' ';
    std::cin >> c;
    if (c == p)
      ++r;
    p = c;
  }
  std::cout << r << std::endl;

  return 0;
}
