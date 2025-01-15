#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  cin >> str;

  for (int i = 0; i < str.length(); i++) {
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y' || str[i] == 'A' || str[i] == 'E' ||
        str[i] == 'I' || str[i] == 'O' || str[i] == 'U' || str[i] == 'Y')
      continue;
    else if (str[i] < 97) {
      char ch = str[i] = str[i] + 32;
      cout << "." << ch;
    } else
      cout << "." << str[i];
  }

  return 0;
}
