#include <stdio.h>

int main() {
  int t;
  char array[51];
  int count = 0;

  scanf("%d", &t);
  getchar();
  for (int i = 0; i < t; i++) {
    scanf("%c", &array[i]);
    if (i != 0) {
      if (array[i] == array[i - 1])
        count++;
    }
  }

  printf("%d\n", count);

  return 0;
}
