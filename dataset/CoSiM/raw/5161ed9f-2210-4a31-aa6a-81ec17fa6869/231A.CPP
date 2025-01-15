/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Team
Problem Link : http://codeforces.com/problemset/problem/231/A
*/

#include <bits/stdc++.h>
using namespace std;

int main() {

  int T, cnt = 0;
  cin >> T;

  while (T--) {

    int a, b, c;
    cin >> a >> b >> c;

    if ((a == 1 && b == 1) || (a == 1 && c == 1) || (b == 1 && c == 1) ||
        (a == 1 && b == 1 && c == 1)) {
      cnt++;
    }
  }

  cout << cnt << endl;
  // system("pause");
  return 0;
}
