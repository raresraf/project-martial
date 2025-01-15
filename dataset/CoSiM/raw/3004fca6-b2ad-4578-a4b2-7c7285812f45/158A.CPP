/*
    Problem Link    :   https://codeforces.com/problemset/problem/158/A
    Contest Link    :   https://codeforces.com/contest/158/problem/A
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
  int n, k;
  cin >> n >> k;
  vector<int> a;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    a.push_back(temp);
  }
  int c = a[k - 1];
  int count = 0;
  for (int x : a) {
    if (x >= c && x > 0)
      count++;
  }
  cout << count << endl;
  return 0;
}