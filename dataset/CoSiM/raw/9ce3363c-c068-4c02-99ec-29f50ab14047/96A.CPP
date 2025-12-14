#include <cmath>
#include <iostream>
#include <string>

using namespace std;

int f(string &s) {
  int mx = 1, cnt = 1;
  for (int i = 1; i <= s.length(); i++) {
    if (s[i] == s[i - 1])
      cnt++;
    else
      cnt = 1;
    if (cnt > mx)
      mx = cnt;
  }
  return mx;
}

void solve() {
  string s;
  cin >> s;
  if (f(s) >= 7)
    cout << "YES\n";
  else
    cout << "NO\n";
}

int main() {
  solve();
  return 0;
}