#include <iostream>
using namespace std;
int main() {
  char s;
  while (cin >> s) {
    if (s == 'A' || s == 'a' || s == 'O' || s == 'o' || s == 'Y' || s == 'y' ||
        s == 'E' || s == 'e' || s == 'U' || s == 'u' || s == 'I' || s == 'i')
      continue;
    else {
      if (s >= 'B' && s <= 'Z')
        s += 32;
      cout << '.' << s;
    }
  }
  return 0;
}
