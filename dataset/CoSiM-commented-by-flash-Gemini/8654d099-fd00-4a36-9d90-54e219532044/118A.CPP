#include <iostream>
#include <string>
using namespace std;

int main() {
  int i;
  string s;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    char x = tolower(s[i]);
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'y') {
      continue;
    } else
      cout << "." << x;
  }
  return 0;
}