#include <bits/stdc++.h>
using namespace std;

int main() {
  int i = 0, c = 0;
  string s, ct;
  cin >> s;
  while (s[i] != '\0') {
    if (s[i] == '4' || s[i] == '7') {
      c++;
    }
    i++;
  }
  ct = to_string(c);
  i = 0;
  c = 0;
  while (ct[i] != '\0') {
    if (ct[i] != '4' && ct[i] != '7') {
      c = 1;
      break;
    }
    i++;
  }
  if (c == 1)
    cout << "NO";
  else
    cout << "YES";
}
