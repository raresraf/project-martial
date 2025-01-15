#include <iostream>
#include <stdio.h>
using namespace ::std;
main() {
  ios_base::sync_with_stdio(false); // Fast I/O
  string s;
  cin >> s;
  int a[26] = {0};
  int ans = 0;
  int l = s.length();
  for (int i = 0; i < l; i++) {
    if (a[s[i] - 97] == 0) {
      ans++;
      a[s[i] - 97] = 1;
    }
  }
  if (ans % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}