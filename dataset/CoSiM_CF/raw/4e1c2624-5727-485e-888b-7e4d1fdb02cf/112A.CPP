#include <bits/stdc++.h>
#include <iostream>

int main() {
  std::string first_str, second_str;
  std::cin >> first_str >> second_str;
  transform(first_str.begin(), first_str.end(), first_str.begin(), ::tolower);
  transform(second_str.begin(), second_str.end(), second_str.begin(),
            ::tolower);
  int output = 0;
  for (unsigned long long index = 0; index < first_str.length(); index++) {
    if ((int)first_str.at(index) > (int)second_str.at(index)) {
      output = 1;
      break;
    } else if ((int)first_str.at(index) < (int)second_str.at(index)) {
      output = -1;
      break;
    }
  }
  std::cout << output << std::endl;

  return 0;
}
