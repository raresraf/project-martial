/*
    Problem Link    :   https://codeforces.com/problemset/problem/339/A
    Contest Link    :   https://codeforces.com/contest/339/problem/A
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
  string s;
  cin >> s;
  vector<char> v;
  for (char x : s) {
    if (x != '+')
      v.push_back(x);
  }
  sort(v.begin(), v.end());
  string ans = "";
  ans += v[0];
  v.erase(v.begin());

  for (char x : v) {
    ans += '+';
    ans += x;
  }
  cout << ans << endl;
  return 0;
}