//
//  71A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int n;
  vector<string> words;
  string str;

  cin >> n;

  while (n--) {
    cin >> str;
    words.push_back(str);
  }

  for (const string s : words) {
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else
      cout << s << endl;
  }

  return 0;
}
