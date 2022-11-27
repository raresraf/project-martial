// Verdict : Accepted
// OJ      : Codeforces
#include <bits/stdc++.h>
using namespace std;
int main() {
  int w;
  while (scanf("%d", &w) == 1) {
    if (w % 2 == 1 || w == 2)
      printf("NO\n");
    else
      printf("YES\n");
  }
  return 0;
}
