#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int len = s.size();
  int a[4] = {0};
  int count = 0;
  for (int i = 0; i < len; i++) {
    if (s[i] != '+') {
      a[(s[i] - '0')]++;
    }
  }

  int k = 1;
  for (int i = 0; i < len; i++) {
    if (i % 2 == 0) {
      while (a[k] == 0)
        k++;

      cout << k;
      a[k]--;
    } else
      cout << "+";
  }

  return 0;
}
