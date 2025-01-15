/// Problem Name: Petya and Strings
/// Problem Link: https://codeforces.com/problemset/problem/112/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.size(); i++) {
    if (a[i] >= 97)
      a[i] = a[i] - 32;
  }
  for (int i = 0; i < b.size(); i++) {
    if (b[i] >= 97)
      b[i] = b[i] - 32;
  }
  // cout << a << " " << b << '\n';
  int ans = 0;
  if (a < b)
    ans--;
  if (a > b)
    ans++;
  cout << ans << '\n';
  return 0;
}
