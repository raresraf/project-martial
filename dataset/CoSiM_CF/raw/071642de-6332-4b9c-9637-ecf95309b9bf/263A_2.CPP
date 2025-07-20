/*
        Both solutions take 124ms. hmmm
*/
#include <stdio.h>

int main() {
  char ch;
  int i = 0;
  int data[25] = {4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0,
                  1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4};

  while ((ch = getchar()) != '1') {
    if (ch == '0')
      i++;
  }

  printf("%d\n", data[i]);
}