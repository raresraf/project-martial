#include <iostream>
using namespace std;
int main() {
  char str[100];
  int i;
  cin >> str;
  for (i = 0; str[i] != '\0'; i++) {
    if (str[i] >= 65 && str[i] <= 90)
      str[i] += 32;
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y') {

    } else
      cout << "." << str[i];
  }
  return 0;
}
