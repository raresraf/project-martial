#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;

  bool h = false, e = false, l = false, l2 = false, o = false;
  int iDunnoWhat = 0;

  for (int i = 0; i < s.size(); i++) {
    if (s.at(i) == 'h')
      h = true;
    if (h && s.at(i) == 'e')
      e = true;
    if (e && s.at(i) == 'l' && !l)
      l = true;
    else if (l && s.at(i) == 'l')
      l2 = true;
    if (l2 && s.at(i) == 'o')
      o = true;
  }

  if (o)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
