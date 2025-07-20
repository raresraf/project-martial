#include <iostream>
using namespace std;
int main() {

  string txt;
  cin >> txt;
  for (int i = 0; i < txt.length(); i++) {
    switch (txt[i]) {
    case 'a':
    case 'A':
    case 'e':
    case 'E':
    case 'i':
    case 'I':
    case 'o':
    case 'O':
    case 'u':
    case 'U':
    case 'y':
    case 'Y':
      break;
    default:
      cout << "." << char(tolower(txt[i]));
    }
  }

  return 0;
}
