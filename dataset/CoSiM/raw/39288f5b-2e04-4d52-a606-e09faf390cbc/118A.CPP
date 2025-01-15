#include <ctype.h>
#include <iostream>

using namespace std;

int main() {
  char s[101];
  char aux;
  int i = 0;

  cin >> s;

  while (s[i]) {
    s[i] = tolower(s[i]);

    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      i++;
      continue;
    } else {
      cout << "." << s[i];
      i++;
    }
  }

  cout << endl;

  return 0;
}