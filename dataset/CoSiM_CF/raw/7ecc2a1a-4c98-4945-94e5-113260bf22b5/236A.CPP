/*
    Problem Link    :   https://codeforces.com/problemset/problem/236/A
    Contest Link    :   https://codeforces.com/contest/236/problem/A
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
  set<char> a;
  for (char x : s)
    a.insert(x);
  if (a.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}