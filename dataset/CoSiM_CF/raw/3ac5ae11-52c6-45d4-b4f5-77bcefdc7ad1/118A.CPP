#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, s1;
  int k = 0;
  cin >> s;
  int l = s.length();
  for (int i = 0; i < l; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'o' || s[i] == 'u' ||
        s[i] == 'i' || s[i] == 'A' || s[i] == 'E' || s[i] == 'O' ||
        s[i] == 'I' || s[i] == 'U') {
      continue;
    } else {
      if (s[i] >= 'A' && s[i] <= 'Z') {
        s1[k] = s[i] + 32;
        k++;
      } else {
        s1[k] = s[i];
        k++;
      }
    }
  }
  for (int i = 0; i < k; i++) {
    if (i <= k - 1)
      cout << ".";
    cout << s1[i];
  }
}
