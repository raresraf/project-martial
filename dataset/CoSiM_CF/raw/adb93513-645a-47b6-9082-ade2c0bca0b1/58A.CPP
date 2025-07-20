#include <bits/stdc++.h>

using namespace std;
int main() {
  string s;
  int i, a[6], j = 0, x, cnt = 0;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (s[i] == 'h') {
      a[j] = i;
      x = i;
      j++;
      break;
    }
  }
  for (i = x + 1; i < s.length(); i++) {
    if (s[i] == 'e') {
      a[j] = i;
      x = i;
      j++;
      break;
    }
  }
  for (i = x + 1; i < s.length(); i++) {
    if (s[i] == 'l') {
      a[j] = i;
      x = i;
      j++;
      break;
    }
  }
  for (i = x + 1; i < s.length(); i++) {
    if (s[i] == 'l') {
      a[j] = i;
      x = i;
      j++;
      break;
    }
  }
  for (i = x + 1; i < s.length(); i++) {
    if (s[i] == 'o') {
      a[j] = i;
      x = i;
      j++;
      break;
    }
  }

  j == 5 ? cout << "YES\n" : cout << "NO\n";

  return 0;
}
