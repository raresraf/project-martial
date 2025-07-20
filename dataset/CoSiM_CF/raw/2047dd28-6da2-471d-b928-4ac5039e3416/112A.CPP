//
//  112A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

#include <iostream>
#include <string>

using namespace std;

int main() {
  string s1, s2;
  int sum_s1 = 0, sum_s2 = 0;

  cin >> s1;

  for (int i = 0; i < s1.length(); ++i) {
    s1[i] = tolower(s1[i]);
    sum_s1 += (s1[i] - '0');
  }

  cin >> s2;

  for (int i = 0; i < s2.length(); ++i) {
    s2[i] = tolower(s2[i]);
    sum_s2 += (s2[i] - '0');
  }

  if (s1 < s2)
    cout << "-1";
  else if (s2 < s1)
    cout << "1";
  else
    cout << "0";
}
