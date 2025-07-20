#include <bits/stdc++.h>

using namespace std;

void solve() {
  string s;
  cin >> s;
  int count = 0;

  sort(s.begin(), s.end());

  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] != s[i + 1])
      count++;
  }

  if (count % 2) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
}

int main() {
  solve();
  return 0;
}
