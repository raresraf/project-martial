#include <bits/stdc++.h>
using namespace std;
main() {
  string s;
  cin >> s;
  int a[100000];
  int j = 0;
  for (int i = 0; i < s.length(); i++) {
    if (isdigit(s[i])) {
      int k = s[i];
      k = k - 48;
      a[j] = k;
      j++;
    }
  }

  sort(a, a + j);
  for (int i = 0; i < j; i++) {
    if (i == 0) {
      cout << a[i];
    } else
      cout << "+" << a[i];
  }
}
