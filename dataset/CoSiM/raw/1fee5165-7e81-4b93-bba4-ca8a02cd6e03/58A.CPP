#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int flagh = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'h' && flagh == 0)
      flagh = 1;
    else if (flagh == 1 && s[i] == 'e')
      flagh = 2;
    else if (flagh == 2 && s[i] == 'l')
      flagh = 3;
    else if (flagh == 3 && s[i] == 'l')
      flagh = 4;
    else if (flagh == 4 && s[i] == 'o')
      flagh = 5;
  }
  if (flagh == 5)
    cout << "YES";
  else
    cout << "NO";
}