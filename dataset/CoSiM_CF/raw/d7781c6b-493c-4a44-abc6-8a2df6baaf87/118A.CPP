#include <algorithm>
#include <cstdio>
#include <ctype.h>
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  string str, str2;
  cin >> str;
  int i, len = str.size();
  for (i = 0; i < len; i++) {
    str[i] = tolower(str[i]);
    if (str[i] != 'a' && str[i] != 'e' && str[i] != 'i' && str[i] != 'o' &&
        str[i] != 'u' && str[i] != 'y')
      str2 += str[i];
  }
  len = str2.size();
  str = "";
  for (i = 0; i < len; i++) {
    str += '.';
    str += str2[i];
  }
  cout << str;
  return 0;
}
