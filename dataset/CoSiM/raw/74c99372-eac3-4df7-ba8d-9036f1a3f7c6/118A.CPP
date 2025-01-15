// 118A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s, res;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'Y' || s[i] == 'A' ||
        s[i] == 'E' || s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
      // do nothing
    } else {
      if (s[i] >= 65 && s[i] <= 90) {
        s[i] = s[i] + 32;
        res.push_back('.');
        res.push_back(s[i]);
      } else {
        res.push_back('.');
        res.push_back(s[i]);
      }
    }
  }
  cout << res << endl;
  return 0;
}
