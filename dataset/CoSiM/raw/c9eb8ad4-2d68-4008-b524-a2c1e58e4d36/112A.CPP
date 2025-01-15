//============================================================================
// Name        : 112A.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int compareStr(string s1, string s2) {
  for (int i = 0; i < s1.length(); i++) {
    if (tolower(s1[i]) < tolower(s2[i])) {
      return -1;
    } else if (tolower(s1[i]) > tolower(s2[i])) {
      return 1;
    }
  }
  return 0;
}

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  cout << compareStr(s1, s2);
}
