#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int i, c = 0, l = s.length();
  for (i = 0; i < l; i++) {
    if (s[i] == '4' || s[i] == '7')
      c++;
  }
  // cout<<"\n"<<c<<"\n";

  if (c == 0)
    cout << "NO";
  else {
    s = to_string(c);
    l = s.length();
    int f = 0;
    for (i = 0; i < l; i++) {
      if (s[i] != '4' && s[i] != '7') {
        f = 1;
        break;
      }
    }
    if (f == 0)
      cout << "YES";
    else
      cout << "NO";
  }
}