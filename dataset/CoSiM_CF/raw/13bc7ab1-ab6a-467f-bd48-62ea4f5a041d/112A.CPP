#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

using namespace std;

char toggleCase(char c) {
  int dif = 'a' - 'A';
  if (c <= 'Z')
    c += dif;
  else
    c -= dif;
  return c;
}

int main() {
  string s, a;
  cin >> s >> a;
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
    a[i] = tolower(a[i]);
  }
  if (s < a)
    cout << -1;
  else if (s > a)
    cout << 1;
  else
    cout << 0;
  return 0;
}