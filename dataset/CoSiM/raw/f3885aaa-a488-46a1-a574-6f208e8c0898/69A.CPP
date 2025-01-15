#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, ans = 0;
  scanf("%d", &n);
  while (n--) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    ans += a + b + c;
  }
  if (ans == 0)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
