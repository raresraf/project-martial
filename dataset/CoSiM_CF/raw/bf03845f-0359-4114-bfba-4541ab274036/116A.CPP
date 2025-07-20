#include <stdio.h>

int main(void) {

  int n;
  int now = 0;
  int capacity = 0;
  int arv;
  int ent;

  scanf("%d", &n);

  for (int i = 0; i <= n; i++) {
    scanf("%d %d", &arv, &ent);

    now = now - arv + ent;
    if (now > capacity) {
      capacity = now;
    }
  }
  printf("%d", capacity);

  return 0;
}
