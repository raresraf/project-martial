#include <bits/stdc++.h>

using namespace std;

int main() {
  int a[1000], flag = 0;
  string s;

  cin >> s;
  int len = s.length();

  for (int i = 0; i < len; i++) {
    if (s[i] == '+')
      continue;
    else
      a[flag++] = s[i] - '0';
  }
  sort(a, a + flag);

  for (int i = 0; i < flag; i++) {
    cout << a[i];
    if (i == flag - 1)
      break;
    cout << "+";
  }

  return 0;
}
