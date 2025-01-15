/*
  Contest 236
  Problem A : Boy or girl
  Author : Rakesh Kumar
  Date: 21/01/2020
*/

#include <bits/stdc++.h>

int main() {
  std::string str;
  std::getline(std::cin, str);

  std::vector<bool> d(26, 0);
  for (char c : str)
    d[c - 'a'] = true;

  int c = 0;
  for (bool i : d)
    if (i)
      ++c;

  printf("%s\n", c & 1 ? "IGNORE HIM!" : "CHAT WITH HER!");
  return 0;
}
