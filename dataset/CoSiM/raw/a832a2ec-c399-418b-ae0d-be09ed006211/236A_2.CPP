#include <bits/stdc++.h>
using namespace std;
int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int l, count = 0, i;
  string s, z;
  cin >> s;
  l = s.length();
  sort(s.begin(), s.end());
  for (i = 1; i < l; i++) {
    if (s[i] != s[i - 1]) {
      count++;
    }
  }
  if ((count + 1) % 2 == 0) {
    printf("CHAT WITH HER!\n");
    cout << s << " " << count;
  } else {
    printf("IGNORE HIM!\n");
    cout << s << " " << count;
  }
  return 0;
}