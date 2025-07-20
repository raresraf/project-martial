#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s;
  cin >> s;

  int pl = s.length() / 2, c(0);
  sort(s.begin(), s.end());
  s.erase(0, pl);

  for (int i = 0; i < pl + 1; i++) {
    cout << s[i];

    if (c != pl)
      cout << "+";

    c++;
  }
}
