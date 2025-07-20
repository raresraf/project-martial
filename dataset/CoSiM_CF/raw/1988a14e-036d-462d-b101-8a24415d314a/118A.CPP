#include <iostream>
#include <string>

using namespace std;

int main() {

  string s;
  cin >> s;
  string result;
  int i;

  for (i = 0; i < s.size(); i++) {

    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      result += "";
    }

    else {

      result += ".";

      if (isupper(s[i])) {
        result += tolower(s[i]);
      } else {
        result += s[i];
      }
    }
  }

  cout << result;
}