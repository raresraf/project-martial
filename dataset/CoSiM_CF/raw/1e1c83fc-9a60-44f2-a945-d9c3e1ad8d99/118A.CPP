#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  string n = "";
  for (int i = 0; i < s.length(); i++) {
    if (s[i] < 'a')
      s[i] += 32;
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i')
      continue;
    else {
      string t = ".";
      t += s[i];
      n += t;
      t = "";
    }
  }
  cout << n << endl;

  return 0;
}
