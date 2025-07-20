/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Bit++
Problem Link : http://codeforces.com/problemset/problem/282/A
*/

#include <bits/stdc++.h>
using namespace std;

int main() {

  int T, x = 0;
  cin >> T;

  while (T--) {

    string s;
    cin >> s;

    if (s[0] == '+' || s[2] == '+') {
      x = x + 1;
    } else {
      x = x - 1;
    }
  }

  cout << x << endl;

  // system("pause");
  return 0;
}
