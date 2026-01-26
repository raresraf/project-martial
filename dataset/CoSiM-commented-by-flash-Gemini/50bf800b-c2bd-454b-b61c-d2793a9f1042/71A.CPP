/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Way Too Long Words
Problem Link : http://codeforces.com/problemset/problem/71/A
*/

#include <bits/stdc++.h>
using namespace std;

int main() {

  int T;
  cin >> T;

  while (T--) {

    string s;
    cin >> s;

    if (s.size() <= 10) {
      cout << s << endl;
    } else {
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    }
  }

  // system("pause");
  return 0;
}
