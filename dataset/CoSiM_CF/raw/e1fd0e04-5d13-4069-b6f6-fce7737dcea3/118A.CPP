#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    s[i] = tolower(s[i]);
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y')
      cout << "." << s[i];
  }
  cout << endl;
  return 0;
}
