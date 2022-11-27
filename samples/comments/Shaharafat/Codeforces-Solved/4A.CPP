#include <bits/stdc++.h>
int main() {
  int i, j, l = 0;
  scanf("%d", &i);
  for (j = 1; j < i; j++) {
    if ((j % 2) == 0) {
      int k = i - j;
      if ((k % 2) == 0) {
        printf("YES\n");
        l = 1;
        break;
      }
    }
  }
  if (l == 0)
    printf("NO\n");
  return 0;
}
