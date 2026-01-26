#include <bits/stdc++.h>
using namespace std;

int main() {
  string t = "hello";
  string s;
  cin >> s;
  int j = 0;
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] == t[j]) {
      t[j++] = 0;
    }
  }
  if (t[t.length() - 1] == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
