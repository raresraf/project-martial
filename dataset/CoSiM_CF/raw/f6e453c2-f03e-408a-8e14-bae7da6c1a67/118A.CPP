#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s, s2;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'a' or s[i] == 'e' or s[i] == 'i' or s[i] == 'o' or
        s[i] == 'u' or s[i] == 'y' or s[i] == 'A' or s[i] == 'E' or
        s[i] == 'I' or s[i] == 'O' or s[i] == 'U' or s[i] == 'Y') {
      continue;
    } else {
      s2 += '.';
      s2 += towlower(s[i]);
    }
  }
  cout << s2;
  return 0;
}
