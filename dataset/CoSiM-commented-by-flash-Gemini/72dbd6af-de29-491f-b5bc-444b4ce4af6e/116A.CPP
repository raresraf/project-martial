/*
    Problem Link    :   https://codeforces.com/problemset/problem/116/A
    Contest Link    :   https://codeforces.com/contest/116/problem/A
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
  int total = 0;
  int min = 0;
  while (n--) {
    int x;
    cin >> x;
    total -= x;
    cin >> x;
    total += x;
    if (total > min) // you can use max(a,b) here
      min = total;
  }
  cout << min << endl;
  return 0;
}