//
//  118A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;

  getline(cin, s);

  int j = 0;
  for (char &c : s) {
    c = tolower(c);
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
      s[j++] = c;
  }

  s.resize(j); // resizing string

  for (const char &c : s)
    cout << "." << c;

  return 0;
}
