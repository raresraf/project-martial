#include <iostream>
using namespace std;
int main() {
  string s, ss = "";
  cin >> s;
  int len = s.length();
  for (int i = 0; i < len; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y')
      continue;
    ss += '.';
    if (s[i] >= 'a' && s[i] <= 'z')
      ss += s[i];
    else
      ss += (s[i] - 'A' + 'a');
  }
  cout << ss;
}