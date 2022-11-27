/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : WaterMelon
Problem Link : http://codeforces.com/problemset/problem/4/A
*/

#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, c = 0;
  cin >> n;

  for (int i = 2; i <= n; i += 2) {
    for (int j = 2; j <= n; j += 2) {
      if (i + j == n) {
        c = 1;
        break;
      }
    }
  }

  cout << ((c == 1) ? "YES" : "NO") << endl;

  // system("pause");
  return 0;
}
