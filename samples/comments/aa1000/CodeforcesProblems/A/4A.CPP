#include "stdio.h"

int main() {
  int w;
  scanf("%d", &w);

  // if the weight is even an greater than two then it can be divided into two
  // even parts
  // because the sum of any two even numbers is even
  if (w > 2 && w % 2 == 0) {
    printf("YES");
  } else
    printf("NO");

  return 0;
}