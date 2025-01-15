#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  int i = 0, n = 0;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    n = s[i] == s[i + 1] ? n + 1 : 0;
    if (n >= 6) {
      cout << "YES"
           << "\n";
      break;
    }
    if (i == s.length() - 1 && n < 6) {
      cout << "NO"
           << "\n";
      break;
    }
  }
  return 0;
}
