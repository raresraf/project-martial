/*
    Problem Link    :   https://codeforces.com/problemset/problem/546/A
    Contest Link    :   https://codeforces.com/contest/546/problem/A
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
  long long k, n, w;
  cin >> k >> n >> w;
  long long sum = ((w * (w + 1)) / 2) * k;
  if (n >= sum)
    cout << 0 << endl;
  else
    cout << sum - n << endl;
  return 0;
}