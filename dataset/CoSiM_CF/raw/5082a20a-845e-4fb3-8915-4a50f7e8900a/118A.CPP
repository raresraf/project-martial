#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); ++i) {
    switch (tolower(s[i])) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
      break;
    default:
      cout << '.' << (char)tolower(s[i]);
    }
  }
  return 0;
}