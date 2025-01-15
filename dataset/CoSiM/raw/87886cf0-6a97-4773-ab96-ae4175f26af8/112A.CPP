#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
int main() {
  char s1[200], s2[200];
  cin >> s1;
  cin >> s2;
  int len = strlen(s1);

  for (int i = 0; i < len; i++) {
    s1[i] = (char)tolower(s1[i]);
    s2[i] = (char)tolower(s2[i]);
  }

  int ans = strcmp(s1, s2);
  cout << ans;

  return 0;
}
