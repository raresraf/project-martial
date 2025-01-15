#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  putchar(toupper(str[0]));
  for (int i = 1; i < str.length(); i++) {
    std::cout << str[i];
  }
  return 0;
}