//============================================================================
// Name        : 71A.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  string s;
  while (n--) {
    cin >> s;
    int z = s.size();
    if (z > 10)
      cout << s[0] << z - 2 << s[z - 1] << endl;
    else
      cout << s << endl;
  }
  return 0;
}
