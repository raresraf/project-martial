#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] <= 90)
      s[i] += 32;
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      continue;
    } else
      cout << "." << s[i];
  }
}
