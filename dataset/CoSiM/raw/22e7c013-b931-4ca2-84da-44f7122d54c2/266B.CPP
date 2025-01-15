#include <cstdio>

int n, t;
char q[51];

int main() {
  scanf("%d%d", &n, &t);

  scanf("%s", q);

  for (int i = 0; i < t; i++) {
    for (int j = 1; j < n; j++) {
      if (q[j] == 'G' && q[j - 1] == 'B') {
        q[j] = 'B';
        q[j - 1] = 'G';
        j++;
      }
    }
  }

  printf("%s", q);
  return 0;
}