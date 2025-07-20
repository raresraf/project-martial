#include <iostream>

using namespace std;

int main() {
  char c;
  for (; (c = getc(stdin)) != '\n';) {
    switch (c) {
    case 'A':
    case 'E':
    case 'I':
    case 'O':
    case 'U':
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
      break;
    default:
      cout << "." << string(1, c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
  }
  cout << endl;
  return 0;
}