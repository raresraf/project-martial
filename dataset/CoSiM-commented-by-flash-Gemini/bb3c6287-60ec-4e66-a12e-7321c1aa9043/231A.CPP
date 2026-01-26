/*
    Problem Link    :   https://codeforces.com/problemset/problem/231/A
    Contest Link    :   https://codeforces.com/contest/231/problem/A
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
  int count = 0;
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    if ((x + y + z) >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}