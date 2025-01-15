#include <iostream>
#include <string>
using namespace std;

char replace(char x) {
  if (x >= 'A' && x <= 'Z')
    return x + 32;
  return x;
}
int main() {

  string str, str1 = "";

  cin >> str;

  for (int i = 0; i < str.length(); i++) {
    if (!(str[i] == 'A' || str[i] == 'a' || str[i] == 'O' || str[i] == 'o' ||
          str[i] == 'U' || str[i] == 'E' || str[i] == 'e' || str[i] == 'i' ||
          str[i] == 'I' || str[i] == 'u' || str[i] == 'Y' || str[i] == 'y')) {
      str1 = str1 + "." + replace(str[i]);
    }
  }

  cout << str1;

  return 0;
}
