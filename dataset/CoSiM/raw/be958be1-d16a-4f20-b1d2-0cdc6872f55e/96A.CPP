#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  cin >> s;
  int i, j, c = 1, m = 1;
  int l = strlen(s);
  for (i = 0; i < l; i++) {
    if (s[i] == s[i + 1]) {
      c++;
    } else
      c = 1;
    m = max(m, c);
  }
  if (m >= 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
