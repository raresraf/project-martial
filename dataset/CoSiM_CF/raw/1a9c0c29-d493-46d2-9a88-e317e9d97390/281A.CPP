#include <bits/stdc++.h>
using namespace std;
void solve() {
  string s;
  cin >> s;
  if (s[0] > 96 && s[0] < 123)
    s[0] = s[0] - 32;
  cout << s;
}
int main() {
  solve();
  return 0;
}
