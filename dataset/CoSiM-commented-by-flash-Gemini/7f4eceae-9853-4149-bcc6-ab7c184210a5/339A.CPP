#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, j;
  char temp;
  string s;
  cin >> s;
  for (i = 0; i < s.size(); i++) {
    for (j = i + 1; j < s.size(); j++) {
      if (s[i] != '+' && s[j] != '+' && s[j] < s[i]) {
        temp = s[j];
        s[j] = s[i];
        s[i] = temp;
      }
    }
  }
  cout << s;

  return 0;
}
