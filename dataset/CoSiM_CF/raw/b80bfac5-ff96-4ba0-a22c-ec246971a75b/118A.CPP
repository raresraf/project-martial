#include <cctype> //for isupper() and tolower()
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int len, i;
  char str[102];
  cin >> str;
  len = strlen(str);
  for (i = 0; i < len; i++) {

    // Condition for vowel ..I don't fucking know any other way to do this
    if (str[i] == 'A' || str[i] == 'O' || str[i] == 'Y' || str[i] == 'E' ||
        str[i] == 'U' || str[i] == 'I' || str[i] == 'a' || str[i] == 'o' ||
        str[i] == 'y' || str[i] == 'e' || str[i] == 'u' || str[i] == 'i') {
      continue;
    } else {
      cout << ".";

      cout << char(tolower(str[i]));
    }
  }
  cout << "\n";
  return 0;
}