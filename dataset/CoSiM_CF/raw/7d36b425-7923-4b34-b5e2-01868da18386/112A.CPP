/*
    Problem Link    :   https://codeforces.com/problemset/problem/112/A
    Contest Link    :   https://codeforces.com/contest/112/problem/A
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
  string a, b;
  cin >> a >> b;
  for (auto &c : a)
    c = tolower(c);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  cout << a.compare(b) << endl;
  return 0;
}