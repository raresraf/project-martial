#include <bits/stdc++.h>
using namespace std;

int main() {
  int z = 0, o = 1;
  string s;
  cin >> s;
  int i = 0;
  while (s[i] != '\0') {
    if (s[i] == '0') {
      z++;
      o = 0;
      if (z == 7)
        break;
    } else {
      o++;
      z = 0;
      if (o == 7)
        break;
    }
    i++;
  }
  if (z == 7 || o == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
