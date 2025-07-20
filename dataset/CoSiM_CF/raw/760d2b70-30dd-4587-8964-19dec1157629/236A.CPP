#include <iostream>

using namespace std;

int main() {

  string s;
  string str = " ";
  int f = 1;

  getline(cin, s);

  for (int i = 0; i < s.length(); ++i) {
    for (int j = 0; j < str.length(); ++j) {
      if (s[i] != str[j]) {
        f = 0;
      } else {
        f = 1;
        break;
      }
    }
    if (f == 0) {
      str += s[i];
    }
    f = 1;
  }

  if ((str.length() - 1) % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }

  return 0;
}
