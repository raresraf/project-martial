#include <iostream>

using namespace std;

bool danger(string s) {
  int flag = 0;
  static int count = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == s[i + 1]) {
      count++;
    } else
      count = 0;
    if (count == 6) {
      flag = 1;
      break;
    }
  }
  return flag;
}

int main() {
  string st;
  while (cin >> st) {
    if (danger(st))
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }

  return 0;
}
