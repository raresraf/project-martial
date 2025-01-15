#include <cstring>
#include <iostream>

using namespace std;

int main() {
  string s;

  cin >> s;

  char c;

  for (int i = 0; i < s.length(); i += 2) {
    for (int j = 0; j < (s.length() - 1 - i); j += 2) {
      if (s[j] > s[j + 2]) {
        c = s[j];
        s[j] = s[j + 2];
        s[j + 2] = c;
      }
    }
  }

  cout << s << endl;

  return 0;
}
