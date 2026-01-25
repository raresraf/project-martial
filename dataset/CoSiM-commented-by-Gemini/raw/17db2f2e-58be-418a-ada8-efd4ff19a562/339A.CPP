//
//  339A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, out;

  cin >> s;

  for (int i = 0; i < s.length(); ++i)
    if (i % 2 == 0)
      out += s[i];

  sort(out.begin(), out.end());

  for (int i = 0; i < out.length(); ++i) {
    i != out.length() - 1 ? cout << out[i] << '+' : cout << out[i];
  }
}
