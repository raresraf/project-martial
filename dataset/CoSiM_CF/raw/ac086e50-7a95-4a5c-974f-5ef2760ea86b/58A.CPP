#include <bits/stdc++.h>

#include <string.h>
using namespace std;

int main() {
  string s;
  int i = 0;

  cin >> s;

  while (i < s.size() && s[i] != 'h')
    ++i;
  ++i;

  while (i < s.size() && s[i] != 'e')
    ++i;
  ++i;
  while (i < s.size() && s[i] != 'l')
    ++i;
  ++i;

  while (i < s.size() && s[i] != 'l')
    ++i;
  ++i;

  while (i < s.size() && s[i] != 'o')
    ++i;

  if (i < s.size()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
