#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;
int main() {
  int i, j;
  string str, check;
  char ch;
  // freopen("in.txt","r",stdin);
  cin >> str;

  for (i = 0; i < str.length(); i++) {
    for (j = 0; j < check.length(); j++) {
      if (str[i] == check[j]) {
        break;
      }
    }
    if (j == check.length()) {
      check += str[i];
    }
  }
  if (check.length() & 1) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}
