#include <iostream>

using namespace std;

int main() {
  int ln1, ln2, i, k = 0;
  string str1, str2;
  cin >> str1 >> str2;
  ln1 = str1.length();
  ln2 = str2.length();

  if (ln1 == ln2) {
    for (i = 0; i < ln1; i++) {
      if (tolower(str1[i]) > tolower(str2[i])) {
        k = 1;
        break;
      }
      if (tolower(str1[i]) < tolower(str2[i])) {
        k = -1;
        break;
      }
      if (tolower(str1[i]) < tolower(str2[i])) {
        k = 0;
      }
    }
    if (k == 1) {
      cout << "1";
    }
    if (k == -1) {
      cout << "-1";
    }
    if (k == 0) {
      cout << "0";
    }
  }

  return 0;
}