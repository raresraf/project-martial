#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    char c = tolower(s[i]);
    if (c == 'a' || c == 'o' || c == 'u' || c == 'y' || c == 'e' || c == 'i')
      continue;
    else
      cout << "." << c;
  }
  return 0;
}
