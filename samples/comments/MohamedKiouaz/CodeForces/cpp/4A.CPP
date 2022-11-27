#include <bits/stdc++.h>

int main() {
  unsigned short acc = 0;
  scanf("%hu", &acc);
  if (acc % 2 == 0 and acc > 2)
    printf("YES");
  else
    printf("NO");
}
