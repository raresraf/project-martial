#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, out = 0;
  string s;
  scanf("%d", &n);
  cin >> s;
  for (int i = 0; i < n; i++) {
    if (s[i] == s[i + 1])
      out++;
  }
  printf("%d", out);
}
