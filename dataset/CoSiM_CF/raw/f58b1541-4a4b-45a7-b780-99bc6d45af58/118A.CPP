#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  string r = "";
  for (int i = 0; i < s.length(); i++) {
    switch (s[i]) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
      break;
    default:
      r += ".";
      r += s[i];
    }
  }
  cout << r;
}