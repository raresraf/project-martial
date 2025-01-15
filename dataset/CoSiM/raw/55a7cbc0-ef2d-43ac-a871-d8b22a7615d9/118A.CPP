#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  int len = s.length();
  for (int i = 0; i < len; i++) {
    if (s[i] == 'a' || s[i] == 'y' || s[i] == 'Y' || s[i] == 'e' ||
        s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' ||
        s[i] == 'E' || s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
      int i = 0;
    }

    else {
      if (s[i] >= 65 && s[i] <= 90) {
        s[i] = s[i] + 32;
        cout << "." << s[i];
      }

      else
        cout << "." << s[i];
    }
  }

  return 0;
}
