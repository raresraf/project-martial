#include <iostream>
#include <string>
using namespace std;

char toLowercase(char c) {
  if (c >= 'A' && c <= 'Z')
    return c + 32;
  return c;
}

int main() {
  string str, s;
  int x;
  cin >> s >> str;
  for (unsigned int i = 0; i < s.length(); i++) {
    s[i] = toLowercase(s[i]);
    str[i] = toLowercase(str[i]);
  }
  if (s == str)
    cout << 0;
  else {
    x = (s > str) ? 1 : -1;
    cout << x << endl;
  }
  return 0;
}
