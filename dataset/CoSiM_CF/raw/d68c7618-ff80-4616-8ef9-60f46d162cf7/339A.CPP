#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  int hold;
  cin >> s;
  for (int j = 0; j < s.length() / 2; j++) {
    for (int i = 2; i < s.length(); i += 2) {
      if (s[i - 2] > s[i]) {
        hold = s[i - 2];
        s[i - 2] = s[i];
        s[i] = hold;
      }
    }
  }
  cout << s;
  return 0;
}