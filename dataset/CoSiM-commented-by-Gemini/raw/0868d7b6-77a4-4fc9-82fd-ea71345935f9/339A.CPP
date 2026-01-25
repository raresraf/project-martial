//============================================================================
// Name        : 339A.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  multiset<int> v;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      v.insert(1);
    if (s[i] == '2')
      v.insert(2);
    if (s[i] == '3')
      v.insert(3);
  }
  string res = "";
  for (auto it = v.begin(); it != v.end(); it++) {
    res += to_string(*it);
    res += '+';
  }

  for (int i = 0; i < res.length() - 1; i++) {
    cout << res[i];
  }
  return 0;
}
