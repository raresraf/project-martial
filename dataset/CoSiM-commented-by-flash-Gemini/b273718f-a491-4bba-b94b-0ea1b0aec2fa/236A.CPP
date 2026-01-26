#include <iostream>

using namespace std;

int main() {
  int len, i, j, n = 0, f;
  string str;

  cin >> str;
  len = str.size();

  for (i = 0; i < len; i++) {
    f = 0;
    for (j = i + 1; j < len; j++) {
      if (str[i] == str[j]) {
        f = 1;
      }
    }
    if (f == 1)
      n++;
  }
  len -= n;
  if (len % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}