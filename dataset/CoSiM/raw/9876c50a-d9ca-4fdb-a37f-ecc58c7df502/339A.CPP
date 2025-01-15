#include <iostream>
#include <string.h>

using namespace std;

int main() {
  char s[101];
  int u = 0, d = 0, t = 0, num;

  cin >> s;

  num = strlen(s);

  for (int i = 0; i < num; i++) {
    if (s[i] == '1') {
      u++;
    } else if (s[i] == '2') {
      d++;
    } else if (s[i] == '3') {
      t++;
    }
  }

  num = u + d + t;

  while (num > 0) {
    // cout << num;
    while (u != 0) {

      cout << "1";

      num--;

      if (num != 0) {
        cout << "+";
      }
      u--;
    }
    while (d != 0) {

      cout << "2";

      num--;

      if (num != 0) {
        cout << "+";
      }
      d--;
    }
    while (t != 0) {

      cout << "3";

      num--;

      if (num != 0) {
        cout << "+";
      }
      t--;
    }
  }

  return 0;
}