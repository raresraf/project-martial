#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, ans = "";
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (isupper(s[i]))
      s[i] = tolower(s[i]);
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    else
      ans = ans + '.' + s[i];
  }
  cout << ans;
  return 0;
}
