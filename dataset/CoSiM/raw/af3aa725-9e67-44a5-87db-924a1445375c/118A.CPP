//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

bool is_vowel(char c) {
  for (char i : {'a', 'e', 'i', 'o', 'u', 'y'}) {
    if (c == i) {
      return true;
    }
  }
  return false;
}

int main() {
  string word;

  cin >> word;

  for (char c : word) {
    if (!is_vowel(tolower(c))) {
      cout << '.' << (char)tolower(c);
    }
  }
}