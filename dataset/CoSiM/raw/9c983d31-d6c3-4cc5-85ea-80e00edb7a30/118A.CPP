//
// Created by Taro Young on 2020/9/7.
//
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;
  for (char ch : s) {
    if (!(ch == 'a' || ch == 'A' || ch == 'o' || ch == 'O' || ch == 'y' ||
          ch == 'Y' || ch == 'e' || ch == 'E' || ch == 'u' || ch == 'U' ||
          ch == 'i' || ch == 'I')) {
      if (ch >= 'A' && ch <= 'Z')
        cout << '.' << char(ch + 32);
      else
        cout << '.' << ch;
    }
  }
  return 0;
}