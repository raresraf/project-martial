/*
    Problem Link    :   https://codeforces.com/problemset/problem/263/A
    Contest Link    :   https://codeforces.com/contest/263/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[6][6];
  int n, x, y;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> n;
      if (n) {
        x = i;
        y = j;
      }
    }
  }
  cout << (abs(3 - x) + abs(3 - y)) << endl;
  return 0;
}