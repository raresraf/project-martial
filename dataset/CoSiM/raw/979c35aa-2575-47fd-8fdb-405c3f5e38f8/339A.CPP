#include <iostream>
#include <string.h>
using namespace std;

int main() {
  string s;
  int one = 0, two = 0, three = 0;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s.at(i) == '1')
      one++;
    if (s.at(i) == '2')
      two++;
    if (s.at(i) == '3')
      three++;
  }
  // cout << one << two << three << endl;
  if (one != 0) {
    for (int i = 1; i < one; i++)
      cout << "1+";
    if (two == 0 && three == 0)
      cout << 1;
    else
      cout << "1+";
  }
  if (two != 0) {
    for (int i = 1; i < two; i++)
      cout << "2+";
    if (three == 0)
      cout << 2;
    else
      cout << "2+";
  }
  if (three != 0) {
    for (int i = 1; i < three; i++)
      cout << "3+";
    cout << 3;
  }

  return 0;
}
