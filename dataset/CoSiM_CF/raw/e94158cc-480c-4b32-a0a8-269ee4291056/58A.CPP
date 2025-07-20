#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  string s;
  bool h = false, e = false, l1 = false, l2 = false, o = false;
  int li;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'h' && h == false) {
      h = true;
    } else if (s[i] == 'e' && h == true && e == false) {
      e = true;
    } else if (s[i] == 'l' && e == true && l1 == false) {
      l1 = true;
    } else if (s[i] == 'l' && l1 == true && l2 == false)
      l2 = true;
    else if (s[i] == 'o' && l2 == true && o == false) {
      o = true;
    }
  }
  if (o == true) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}