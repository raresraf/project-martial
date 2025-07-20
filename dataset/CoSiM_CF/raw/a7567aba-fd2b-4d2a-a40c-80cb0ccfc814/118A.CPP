//============================================================================
// Name        : 118A.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  string s2 = "";
  for (int i = 0; i < s.size(); i++) {
    if ((int)s2.size() % 2 == 0)
      s2 += '.';
    char ch = tolower(s[i]);
    if (ch != 'a' && ch != 'o' && ch != 'y' && ch != 'e' && ch != 'u' &&
        ch != 'i')
      s2 += ch;
  }
  if ((int)s2.size() % 2 == 0)
    s2 += '.';
  cout << s2.substr(0, s2.size() - 1);
  return 0;
}
