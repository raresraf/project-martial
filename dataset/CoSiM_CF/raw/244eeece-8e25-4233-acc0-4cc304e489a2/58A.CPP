// link : https://codeforces.com/contest/58/problem/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int h = 0, e = 0, l = 0, o = 0;
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] == 'h' && h < 1)
      h++;
    else if (h == 1 && e < 1 && s[i] == 'e')
      e++;
    else if (e == 1 && l < 2 && s[i] == 'l')
      l++;
    else if (l == 2 && s[i] == 'o')
      o++;
  }
  if (h != 0 && e != 0 && l != 0 && o != 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
