/*
    Problem Link    :   https://codeforces.com/problemset/problem/160/A
    Contest Link    :   https://codeforces.com/contest/160/problem/A
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
  vector<int> a(n);
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a.begin(), a.end());
  reverse(a.begin(), a.end());
  int mx = 0;
  int i = 0;
  for (i = 0; i < n; i++) {
    mx += a[i];
    if (mx > sum - mx)
      break;
  }
  cout << i + 1 << endl;
  return 0;
}