#include <bits/stdc++.h>

using namespace std;

int main() {
  bool n[150];
  memset(n, 0, 150);

  string s;
  getline(cin, s);
  int len = s.size();
  int c = 0;
  for (int i = 0; i < len; i++) {
    if (n[s[i]] == 0) {
      n[s[i]] = 1;
      c++;
    }
  }
  if (c % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");

  return 0;
}
