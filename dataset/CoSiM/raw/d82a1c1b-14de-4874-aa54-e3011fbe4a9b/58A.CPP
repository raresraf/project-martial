#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, s1;
  s1 = "hello";
  cin >> s;
  int i, j = 0, c;

  for (i = 0; i < s1.size(); i++) {
    c = 0;
    for (; j < s.size(); j++) {
      if (s[j] == s1[i]) {
        c = 1;
        break;
      }
    }
    j = j + 1;
    if (c == 0) {
      cout << "NO" << endl;
      return 0;
    }
  }
  if (c == 1)
    cout << "YES" << endl;

  return 0;
}
