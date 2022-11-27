// https://codeforces.com/problemset/problem/4/A

#include <bits/stdc++.h>

using namespace std;

string solve(int w) {
  // weight should be even number and equal or greater than 4
  // to be divided to even numbers.
  return w >= 4 && w % 2 == 0 ? "YES" : "NO";
}

int main() {
  int w;
  cin >> w;
  cout << solve(w);
}
