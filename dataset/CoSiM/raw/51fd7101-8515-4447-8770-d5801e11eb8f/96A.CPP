#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  getline(cin, s);
  int len = s.size();
  int c = 1;
  bool f = false;
  for (int i = 1; i < len; i++) {
    if (s[i] == s[i - 1]) {
      c++;
      if (c == 7) {
        f = true;
        break;
      }
    } else {
      c = 1;
      f = false;
    }
  }
  printf("%s\n", f ? "YES" : "NO");

  return 0;
}
