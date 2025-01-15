// Problem Link https://codeforces.com/problemset/problem/1366/A
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  int t;
  long long a, b;
  cin >> t;

  long long l, c;
  for (int i = 0; i < t; i++) {
    cin >> a >> b;
    l = (long long)floor((a + b) / 3.0);
    c = min({l, a, b});
    cout << c << "\n";
  }
}