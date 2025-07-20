/*
    Problem Link    :   https://codeforces.com/problemset/problem/266/A
    Contest Link    :   https://codeforces.com/contest/266/problem/A
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
  int n;
  cin >> n;
  string s;
  cin >> s;
  if (!n) {
    cout << 0 << endl;
    return 0;
  }

  int count = 0;
  for (int i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1])
      count++;
  }
  cout << count << endl;
  return 0;
}