#include <bits/stdc++.h>
using namespace std;
int n[5];
int main() {
  // freopen("input.txt", "r", stdin);
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      n[s[i] - '0']++;
  }
  bool f = false;
  for (int i = 1; i < 3 + 1; i++) {
    while (n[i]--) {
      if (f)
        printf("+");
      f = true;
      printf("%d", i);
    }
  }
  printf("\n");
  return 0;
}
