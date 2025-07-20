#include <bits/stdc++.h>
using namespace std;
string to_lower(string s) {
  int l = s.length();
  for (int i = 0; i < l; i++) {
    if (s[i] <= 90 && s[i] >= 65) {
      s[i] += 32;
    }
  }
  return s;
}
int main() {
  string s2, s1;
  cin >> s1 >> s2;
  s1 = to_lower(s1);
  s2 = to_lower(s2);
  if (s1 < s2)
    cout << "-1\n";
  else if (s1 == s2)
    cout << "0\n";
  else
    cout << "1\n";
}