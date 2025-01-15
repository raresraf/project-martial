#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int l = s.length();
  char prev = s[0];
  int cnt = 1;
  bool ans = false;
  for (int i = 1; i < l; i++) {
    if (s[i] == prev) {
      cnt++;
      if (cnt == 7) {
        ans = true;
        break;
      }
    } else {
      cnt = 1;
      prev = s[i];
    }
  }
  if (ans)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}