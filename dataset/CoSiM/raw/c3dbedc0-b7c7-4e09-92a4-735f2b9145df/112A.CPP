#include <iostream>

using namespace std;

int main() {
  string c, s;
  cin >> c;
  cin >> s;
  int n = 0;
  for (int i = 0; i < c.length(); i++) {
    c[i] = tolower(c[i]);
    s[i] = tolower(s[i]);
  }
  if (c > s) {
    cout << "1";
  } else if (c < s) {
    cout << "-1";
  } else if (c == s) {
    cout << "0";
  }

  return 0;
}
