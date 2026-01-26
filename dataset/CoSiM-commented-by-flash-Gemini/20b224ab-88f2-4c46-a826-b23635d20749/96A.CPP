#include <iostream>
using namespace std;
int main() {
  string s;
  getline(cin, s);
  string out = "NO";
  if (s.length() > 7) {
    int c = 1;
    for (int i = 1; i < s.length(); i++) {
      if (s[i] == s[i - 1])
        c++;
      else
        c = 1;
      if (c == 7) {
        out = "YES";
        break;
      }
    }
  }
  cout << out;
  return 0;
}
