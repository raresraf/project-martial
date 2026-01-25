#include <iostream>

using namespace std;

int main() {
  int len, i, n = 0, m = 0, a = 0, b = 0;
  string str;
  cin >> str;
  len = str.length();
  for (i = 0; i < len; i++) {
    if (str[i] == '0')
      a++;
    if (str[i] == '1')
      b++;
  }
  if (a > 0 && b > 0) {
    for (i = 0; i < len; i++) {
      if (str[i] == '1') {
        n++;
        if (n == 7) {
          cout << "YES";
          return 0;
        }
        if (str[i + 1] == '0') {
          n = 0;
        }
      }

      if (str[i] == '0') {
        m++;
        if (m == 7) {
          cout << "YES";
          return 0;
        }
        if (str[i + 1] == '1') {
          m = 0;
        }
      }
    }
    cout << "NO";
  }
  return 0;
}
