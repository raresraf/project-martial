#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, t;
  int l;
  cin >> s;
  l = s.size();
  for (int i = 0; i < l; i++) {
    if (s[i] == 'A' || s[i] == 'a' || s[i] == 'E' || s[i] == 'e' ||
        s[i] == 'I' || s[i] == 'i' || s[i] == 'O' || s[i] == 'o' ||
        s[i] == 'U' || s[i] == 'u' || s[i] == 'Y' || s[i] == 'y') {
      continue;
    } else {
      t += ".";
      t += (tolower(s[i]));
    }
  }
  cout << t << endl;
  return 0;
}
