#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  int i, j, k, l;
  while (cin >> s) {
    l = s.length();
    j = 0;
    for (i = 0; i < l; i++) {
      if (l < 7) {
        j = 0;
        break;
      } else if (s[i] == s[i + 1] && s[i] == s[i + 2] && s[i] == s[i + 3] &&
                 s[i] == s[i + 4] && s[i] == s[i + 5] && s[i] == s[i + 6]) {
        j = 1;
        break;
      }
    }
    if (j == 1) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
