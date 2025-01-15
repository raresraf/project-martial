/*
  Contest 110
  Problem A : Nearly lucky number
  Author : Rakesh Kumar
  Date: 30/05/2020
*/

#include <bits/stdc++.h>

int main() {
  std::string str;
  std::getline(std::cin, str);
  std::string result = "YES";

  std::unordered_map<char, int> d;
  for (char c : str)
    ++d[c];

  const int t = d['4'] + d['7'];
  if (t == 4 || t == 7)
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}
