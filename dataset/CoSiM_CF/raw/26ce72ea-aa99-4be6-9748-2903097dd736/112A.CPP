#include <iostream>
using namespace std;
int main() {
  string s, c;
  cin >> s >> c;
  int i;
  for (i = 0; s[i] != '\0'; i++) {
    s[i] = tolower(s[i]);
    c[i] = tolower(c[i]);
  }
  if (s.compare(c) < 0) {
    cout << "-1";
  } else if (s.compare(c) > 0) {
    cout << "1";
  } else
    cout << 0;
}
