#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int i = 0, j, n = 0;
  cin >> s;
  while (s[i] != '\0') {
    j = i + 1;
    while (s[j] != '\0') {
      if (s[i] == s[j])
        s[j] = '1';
      j++;
    }
    i++;
  }
  i = 0;
  while (s[i] != '\0') {
    if (s[i] != '1')
      n++;
    i++;
  }
  if (n % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
