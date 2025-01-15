#include <stdio.h>

int main() {
  int t;
  int count = 0;

  scanf("%d", &t);
  for (int i = 0; i < t; i++) {
    int sum = 0;
    for (int j = 0; j < 3; j++) {
      int digit = 0;
      scanf("%d", &digit);
      if (digit == 1)
        sum++;
    }
    if (sum >= 2)
      count++;
    sum = 0;
  }
  printf("%d\n", count);
  return 0;
}
