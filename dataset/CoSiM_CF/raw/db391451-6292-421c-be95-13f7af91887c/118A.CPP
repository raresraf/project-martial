#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'o' || s[i] == 'i' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    else {
      cout << '.' << s[i];
    }
  }
  cout << endl;
}
