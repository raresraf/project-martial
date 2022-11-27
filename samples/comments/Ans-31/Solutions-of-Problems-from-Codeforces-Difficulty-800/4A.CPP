/// Problem Name: Watermelon
/// Problem Link: https://codeforces.com/problemset/problem/4/A

#include <stdio.h>

int main() {
  int w;
  scanf("%d", &w);
  if (w < 4) {
    printf("NO\n");
  } else {
    if (w % 2 == 0) {
      printf("YES\n");
    } else {
      printf("NO\n");
    }
  }
  return 0;
}
