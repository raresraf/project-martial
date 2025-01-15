#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  int c = 5, i = 0;
  cin >> s;
  for (i = 0; i < s.size(); i++) {
    if (c == 5)
      if (s[i] == 'h')
        c--;
      else
        continue;
    else if (c == 4)
      if (s[i] == 'e')
        c--;
      else
        continue;
    else if (c == 3)
      if (s[i] == 'l')
        c--;
      else
        continue;
    else if (c == 2)
      if (s[i] == 'l')
        c--;
      else
        continue;
    else if (c == 1)
      if (s[i] == 'o') {
        cout << "YES" << endl;
        return 0;
      }
  }
  cout << "NO" << endl;
  return 0;
}
