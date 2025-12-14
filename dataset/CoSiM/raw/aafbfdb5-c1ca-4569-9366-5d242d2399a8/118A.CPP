/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : String Task
Problem Link : http://codeforces.com/problemset/problem/118/A
*/

#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;

  for (int i = 0; i < s.size(); ++i) {
    if (s[i] >= 65 && s[i] <= 90) {
      s[i] = s[i] + 32;
    }
  }

  for (int i = 0; i < s.size(); ++i) {
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y') {
      cout << "." << s[i];
    }
  }

  cout << endl;

  return 0;
}
