#include <bits/stdc++.h>
using namespace std;
main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] != 'a' && s[i] != 'A' && s[i] != 'e' && s[i] != 'E' &&
        s[i] != 'i' && s[i] != 'I' && s[i] != 'o' && s[i] != 'O' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'y' && s[i] != 'y') {
      if (isupper(s[i])) {
        cout << "." << (char)tolower(s[i]);
      } else {
        cout << "." << s[i];
      }
    }
  }
}
