/*
    Problem Link    :   https://codeforces.com/problemset/problem/282/A
    Contest Link    :   https://codeforces.com/contest/282/problem/A
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
  int ans = 0;
  while (n--) {
    string a;
    cin >> a;
    if (a == "++X" || a == "X++")
      ans++;
    else
      ans--;
  }
  cout << ans << endl;
  return 0;
}