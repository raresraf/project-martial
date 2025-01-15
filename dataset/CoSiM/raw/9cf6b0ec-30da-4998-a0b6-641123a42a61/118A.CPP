#include <bits/stdc++.h>
using namespace std;

int main() {
  int i;
  string s;
  cin >> s;
  for (i = 0; i < s.size(); i++) {
    if ((s[i] != 'A') && (s[i] != 'E') && (s[i] != 'I') && (s[i] != 'O') &&
        (s[i] != 'U') && (s[i] != 'Y') && (s[i] != 'a') && (s[i] != 'e') &&
        (s[i] != 'i') && (s[i] != 'o') && (s[i] != 'u') && (s[i] != 'y')) {
      cout << ".";
      if (s[i] >= 'A' && s[i] <= 'Z') {
        s[i] = tolower(s[i]);
        cout << s[i];
      } else
        cout << s[i];
    }
  }
  return 0;
}
