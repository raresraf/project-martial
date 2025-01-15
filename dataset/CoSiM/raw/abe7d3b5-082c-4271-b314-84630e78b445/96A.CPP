/* P96A by Gabriel Scotillo */
#include <iostream>

using namespace std;

int main() {

  string str;
  cin >> str;
  bool isDangerous = false;

  int c0 = 1;
  int c1 = 1;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '0') {
      if (str[i + 1] == '0') {
        c0 = c0 + 1;
        if (c0 == 7) {
          isDangerous = true;
          break;
        }
      } else {
        c0 = 1;
      }
    } else {
      if (str[i + 1] == '1') {
        c1 = c1 + 1;
        if (c1 == 7) {
          isDangerous = true;
          break;
        }
      } else {
        c1 = 1;
      }
    }
  }

  if (isDangerous) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}