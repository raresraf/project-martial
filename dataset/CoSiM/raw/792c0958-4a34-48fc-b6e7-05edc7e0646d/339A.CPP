#include <iostream>
using namespace std;
int main() {
  string s;
  int a = 0, b = 0, c = 0, i;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (s[i] == '1')
      a++;
    else if (s[i] == '2')
      b++;
    else if (s[i] == '3')
      c++;
  }
  while (a--) {
    cout << "1";
    if (a != 0 || b != 0 || c != 0)
      cout << "+";
  }
  while (b--) {
    cout << "2";
    if (b != 0 || c != 0)
      cout << "+";
  }
  while (c--) {
    cout << "3";
    if (c != 0)
      cout << "+";
  }
  cout << endl;
}