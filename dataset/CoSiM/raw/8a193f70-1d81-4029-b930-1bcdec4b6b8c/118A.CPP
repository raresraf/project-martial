#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int i;
  char str[100];
  cin >> str;
  // string processing
  for (i = 0; i < strlen(str); i++) {
    if ((str[i] == 'A') || (str[i] == 'O') || (str[i] == 'Y') ||
        (str[i] == 'E') || (str[i] == 'U') || (str[i] == 'I') ||
        (str[i] == 'a') || (str[i] == 'o') || (str[i] == 'y') ||
        (str[i] == 'e') || (str[i] == 'u') || (str[i] == 'i'))
      str[i] = '-';
    else if ((64 < (int)str[i]) && ((int)str[i] < 91))
      str[i] = (char)((int)str[i] + 32);
  }
  // string printing
  for (i = 0; i < strlen(str); i++) {
    if (str[i] != '-')
      cout << "." << str[i];
  }
  cout << endl;
  return 0;
}
