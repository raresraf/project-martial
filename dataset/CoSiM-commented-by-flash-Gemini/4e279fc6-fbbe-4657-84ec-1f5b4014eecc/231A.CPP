#include <stdio.h>

int main(void) {

  int n;
  int cnt = 0;
  int P[1000];
  int V[1000];
  int T[1000];

  scanf("%d", &n);

  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &P[i], &V[i], &T[i]);
  }

  for (int i = 0; i < n; i++) {
    if (P[i] + V[i] + T[i] >= 2) {
      cnt++;
    }
  }

  printf("%d", cnt);

  return 0;
}
