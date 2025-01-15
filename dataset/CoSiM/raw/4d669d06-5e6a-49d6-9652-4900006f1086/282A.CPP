//
//  282A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>

using namespace std;

int main() {
  int count = 0;
  string s;

  cin >> s;

  for (const char &c : s) {
    if (int(c) >= int('A') && int(c) <= int('Z'))
      count++;
  }

  if ((count == s.length() - 1 && (s[0] >= int('a') && s[0] <= int('z')))) {
    int i = 0;
    for (const char &c : s) {
      i == 0 ? s[i] = toupper(c) : s[i] = tolower(c);
      ++i;
    }
  } else if (count == s.length()) {
    int i = 0;
    for (const char &c : s) {
      s[i] = tolower(c);
      ++i;
    }
  }

  cout << s << "\n";

  return 0;
}
