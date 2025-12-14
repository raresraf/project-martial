#include <bits/stdc++.h>
using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string s, s1;
  cin >> s;
  int l = s.length();
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  for (int i = 0; i < l; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    else {
      s1.push_back('.');
      s1.push_back(s[i]);
    }
  }
  cout << s1 << "\n";
  return 0;
}