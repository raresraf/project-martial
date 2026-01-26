#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int l = s.size();

  for (int i = 0; i < l; i++) {
    for (int j = i + 1; j < l; j++) {
      if (s[i] > s[j]) {
        s[i] = s[i] + s[j];
        s[j] = s[i] - s[j];
        s[i] = s[i] - s[j];
      }
    }
  }
  for (int i = 0; i < l; i++) {
    if (i == l - 1)
      cout << s[i];
    else if (s[i] != '+')
      cout << s[i] << "+";
  }

  return 0;
}