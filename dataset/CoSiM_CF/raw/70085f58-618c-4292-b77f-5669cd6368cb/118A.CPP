#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int l = s.length();
  for (int i = 0; i < l; i++) {
    s[i] = tolower(s[i]);
    if (!(s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
          s[i] == 'u' || s[i] == 'i'))
      cout << "." << s[i];
  }
  cout << "\n";
  return 0;
}