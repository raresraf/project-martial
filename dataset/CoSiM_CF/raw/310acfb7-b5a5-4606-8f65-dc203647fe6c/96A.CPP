#include <iostream>
#include <string>
using namespace std;

int main() {
  long long int c, i;
  string s;
  cin >> s;
  for (i = 0; i < s.length(); i++) {

    if (s[i] == s[i + 1]) {
      c++;
      if (c == 6) {
        cout << "YES";
        break;
      }

    } else {
      c = 0;
    }
  }
  if (c < 6) {
    cout << "NO";
  }
  return 0;
}