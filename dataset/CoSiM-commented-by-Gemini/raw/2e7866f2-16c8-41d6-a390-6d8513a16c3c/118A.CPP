#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int n = 0;
  char d[s.size()];
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != 'A' && s[i] != 'a' && s[i] != 'E' && s[i] != 'e' &&
        s[i] != 'O' && s[i] != 'o' && s[i] != 'Y' && s[i] != 'y' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'I' && s[i] != 'i') {
      d[n] = s[i];
      n++;
    }
  }
  for (int i = 0; i < n; i++) {
    if (d[i] >= 'a' && d[i] <= 'z') {
      cout << "." << d[i];
    } else
      cout << "." << char(d[i] + 32);
  }
  return 0;
}
