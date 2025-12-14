/*
    Problem Link    :   https://codeforces.com/problemset/problem/266/B
    Contest Link    :   https://codeforces.com/contest/266/problem/B
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
  int n, t;
  cin >> n >> t;
  string a;
  cin >> a;
  while (t--) {
    for (int i = 0; i < n - 1; i++)
      if (a[i] == 'B' && a[i + 1] == 'G') {
        swap(a[i], a[i + 1]);
        i++;
      }
  }
  cout << a << endl;
  return 0;
}