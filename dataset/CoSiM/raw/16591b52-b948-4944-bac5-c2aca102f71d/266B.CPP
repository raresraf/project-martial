//============================================================================
// Name        : codeForces_266B.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  for (int i = 0; i < t; i++) {
    for (int j = 1; j < (int)s.size(); j++) {
      if (s[j] == 'G' && s[j - 1] == 'B')
        swap(s[j], s[j - 1]), j++;
    }
  }
  cout << s;
  return 0;
}
