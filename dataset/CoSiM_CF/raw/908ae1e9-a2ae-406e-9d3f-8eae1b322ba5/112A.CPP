/*
  Contest 112
  Problem A : Petya and strings
  Author : Rakesh Kumar
  Date: 17/01/2020
*/

#include <bits/stdc++.h>

const int t = std::abs('a' - 'A');

int main() {
  std::string a;
  std::string b;
  std::getline(std::cin, a);
  std::getline(std::cin, b);

  int r = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int d = std::abs(a[i] - b[i]);
    if (!(d == 0 || d == t)) {
      if (std::tolower(a[i]) < std::tolower(b[i]))
        r = -1;
      else
        r = 1;
      break;
    }
  }

  printf("%d\n", r);

  return 0;
}
