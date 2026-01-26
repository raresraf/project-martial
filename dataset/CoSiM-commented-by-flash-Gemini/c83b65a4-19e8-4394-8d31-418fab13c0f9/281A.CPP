#include <bits/stdc++.h>
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  string str;
  getline(cin, str);
  for (int i = 0; i < str.length(); i++) {
    if (i == 0)
      putchar(toupper(str[i]));
    else
      cout << str[i];
  }

  return 0;
}
