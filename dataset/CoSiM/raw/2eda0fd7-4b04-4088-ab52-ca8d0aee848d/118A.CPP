#include <bits/stdc++.h>
using namespace std;
string s;
string res;
int main() {
  cin >> s;
  for (int i = 0; i < (int)s.length(); i++) {
    if (tolower(s[i]) != 'a' && tolower(s[i]) != 'o' && tolower(s[i]) != 'u' &&
        tolower(s[i]) != 'i' && tolower(s[i]) != 'e' && tolower(s[i]) != 'y') {
      res += '.';
      res += tolower(s[i]);
    }
  }
  cout << res << '\n';
}