#include <cstring>
#include <iostream>
using namespace std;

int main() {
  char str[100];
  int l;
  cin >> str;
  l = strlen(str);
  for (int i = 0; i < l; i++) {
    str[i] = tolower(str[i]);

    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y') {
      continue;
    } else {
      cout << ".";
      cout << str[i];
    }
  }

  return 0;
}
