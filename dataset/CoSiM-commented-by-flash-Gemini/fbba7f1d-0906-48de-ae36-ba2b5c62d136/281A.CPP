/*
  Contest 281
  Problem A : Word capitalization
  Author : Rakesh Kumar
  Date: 18/01/2020
*/

#include <bits/stdc++.h>

int main() {
  std::string str;
  std::getline(std::cin, str);
  if (std::islower(str[0]))
    str[0] -= 32;
  printf("%s\n", str.c_str());

  return 0;
}
