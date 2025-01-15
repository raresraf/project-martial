#include <iostream>
#include <string>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  string s, res;
  cin >> s;
  for (int i = 0; i < (int)s.size(); ++i)
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] - ('A' - 'a');
  for (int i = 0; i < (int)s.size(); ++i) {
    if (s[i] != 'a' && s[i] != 'o' && s[i] != 'y' && s[i] != 'e' &&
        s[i] != 'u' && s[i] != 'i') {
      res += string() + "." + s[i];
    }
  }

  cout << res << endl;
  return 0;
}