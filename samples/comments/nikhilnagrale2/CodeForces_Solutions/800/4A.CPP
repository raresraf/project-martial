/*
    Problem Link    :   https://codeforces.com/problemset/problem/4/A
    Contest Link    :   https://codeforces.com/contest/4/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own. Copy paste is not an option.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  int w;
  cin >> w;
  if (w % 2 == 0 && w > 2)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}