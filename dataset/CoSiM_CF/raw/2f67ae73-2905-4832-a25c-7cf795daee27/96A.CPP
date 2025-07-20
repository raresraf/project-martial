#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  int c = 1;
  for (size_t i = 1; i < s.length(); ++i) {
    if (s[i] == s[i - 1]) {
      c += 1;
      if (c == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      c = 1;
    }
  }

  cout << "NO" << endl;
  return 0;
}
