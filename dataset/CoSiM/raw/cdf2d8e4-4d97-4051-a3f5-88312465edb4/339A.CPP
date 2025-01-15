#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;

  int len = s.length();
  int i = 0;

  for (int i = 0; i < len; i++) {
    for (int j = i + 1; j < len; j++) {
      if (s[i] == '+' || s[j] == '+')
        continue;
      else if (s[i] >= s[j]) {
        char swap = s[i];
        s[i] = s[j];
        s[j] = swap;
      }
    }
  }

  cout << s;
}
