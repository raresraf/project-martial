#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  int i, j;
  cin >> s;
  for (i = 0; i < s.size(); i++) {
    if (s[i] == s[i + 6]) {
      for (j = 1; j < 6; j++) {
        if (j == 5 && s[i + j] == s[i]) {
          cout << "YES";
          return 0;
        } else {
          if (s[i + j] == s[i])
            continue;
          else
            break;
        }
      }
    }
  }
  cout << "NO";
  return 0;
}
