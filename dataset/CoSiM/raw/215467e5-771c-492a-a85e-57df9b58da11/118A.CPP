#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  vector<char> a;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y')
      ;
    else {
      char ch = '.';
      a.push_back(ch);
      ch = s[i];
      if (ch >= 'A' && ch <= 'Z')
        a.push_back(ch - 'A' + 'a');
      else
        a.push_back(ch);
    }
  }
  for (auto &i : a)
    cout << i;
}
