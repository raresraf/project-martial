#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  int i = 0, c = '1', n = 0;

  while (i < s.length()) {
    if (c == s[i]) {
      if (++n == 7)
        break;
    } else {
      n = 1;
      c = s[i];
    }
    i++;
  }
  if (n == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}