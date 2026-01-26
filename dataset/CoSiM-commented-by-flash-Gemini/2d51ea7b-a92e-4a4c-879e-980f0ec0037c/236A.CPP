//
//  236A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main() {
  unordered_map<char, int> h;
  string s;

  cin >> s;

  for (const char c : s) {
    if (h[c] == 1)
      continue;
    else
      h[c] = 1;
  }

  h.size() % 2 == 0 ? cout << "CHAT WITH HER!" : cout << "IGNORE HIM!" << endl;
}
