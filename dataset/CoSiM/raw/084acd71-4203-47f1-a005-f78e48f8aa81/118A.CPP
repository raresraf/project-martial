#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  for (int i = 0; i < str.length(); i++) {
    str[i] = tolower(str[i]);
    if (str[i] == 'a' or str[i] == 'e' or str[i] == 'i' or str[i] == 'o' or
        str[i] == 'u' or str[i] == 'y')
      continue;
    else {
      cout << '.' << str[i];
    }
  }
  return 0;
}