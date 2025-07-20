#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s, n;
  n = "";
  cin >> s;
  //    cout<<s<<endl;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue;
    }

    n = n + '.';
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] + 32;
    n = n + s[i];
  }

  cout << n << endl;
  return 0;
}
