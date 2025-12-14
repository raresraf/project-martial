#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int i, j, k, l;
  string s;
  while (cin >> s) {
    l = s.length();
    for (i = 0; i < l; i++) {
      if (s[i] >= 'a' && s[i] <= 'z' && s[i] != 'a' && s[i] != 'e' &&
          s[i] != 'i' && s[i] != 'o' && s[i] != 'y' && s[i] != 'u') {
        cout << ".";
        cout << s[i];
      } else if (s[i] >= 'A' && s[i] <= 'Z' && s[i] != 'A' && s[i] != 'E' &&
                 s[i] != 'I' && s[i] != 'O' && s[i] != 'Y' && s[i] != 'U') {
        cout << ".";
        cout << (char)tolower(s[i]);
      }
    }
    cout << endl;
  }
  return 0;
}
