#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, s2;
  cin >> s;
  int l = s.length();

  for (int i = 0; i < l; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue;
    }
    s2 += ".";
    s2 += tolower(s[i]);
  }
  return 0;
}