#include <bits/stdc++.h>
#include <cctype>
#include <iostream>
using namespace std;

int main() {
  string str;
  cin >> str;
  if (islower(str[0])) {
    str[0] = toupper(str[0]);
  }
  cout << str;
  return 0;
}
