#include <bits/stdc++.h>
#include <cstring>
using namespace std;
int main() {
  string s, vowels = "aoyeui", rp = "0";
  cin >> s;
  int n = s.length();
  for (int i = 0; i < n; i++) {
    if (vowels.find(tolower(s[i])) >= 0 && vowels.find(tolower(s[i])) <= n) {
      s.replace(i, 1, rp);
    }
  }
  for (int i = 0; i < n; i++) {
    if (s[i] != '0') {
      if (s[i] > 64 && s[i] < 91)
        cout << "." << (char)(s[i] + 32);
      else
        cout << "." << s[i];
    }
  }
  return 0;
}
