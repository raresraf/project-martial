#include <stdio.h>
int main() {
  char str[100], n = 0, m = 0, a, j;
  int i, capacity = 0, sum = 0;
  while (scanf("%s", str) != EOF) {
    for (i = 0; str[i] != '\0'; i++) {
      if (str[i] == '0')
        n++;
      if (n > capacity)
        capacity = n;
      if (str[i] == '1')
        n = 0;
      if (str[i] == '1')
        m++;
      if (m > sum)
        sum = m;
      if (str[i] == '0')
        m = 0;
    }
    if (capacity >= 7 || sum >= 7)
      printf("YES\n");
    else
      printf("NO\n");
  }

  return 0;
}
