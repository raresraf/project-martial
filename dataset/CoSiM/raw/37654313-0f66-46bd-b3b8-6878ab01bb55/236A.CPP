#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  cin >> s;
  int i, j, c = 0;
  int l = strlen(s);
  sort(s, s + l);

  for (i = 0; i < l; i++) {
    if (s[i] != s[i + 1]) {
      c++;
    }
  }
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
}
