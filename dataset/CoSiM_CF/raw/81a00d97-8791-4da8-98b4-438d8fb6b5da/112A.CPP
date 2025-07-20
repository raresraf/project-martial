
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int r = 0;
  string str1, str2;
  cin >> str1 >> str2;
  for (int i = 0; str1[i] != '\0'; i++) {
    if (tolower(str1[i]) > tolower(str2[i])) {
      r = 1;
      break;
    }
    if (tolower(str1[i]) < tolower(str2[i])) {
      r = -1;
      break;
    }
  }
  cout << r;

  return 0;
}
