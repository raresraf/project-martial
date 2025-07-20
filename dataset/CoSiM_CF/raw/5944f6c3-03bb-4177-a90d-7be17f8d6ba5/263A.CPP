#include <iostream>

int main() {
  int v, r = 0;

  for (int i = 0; i < 25; i++) {
    scanf("%d", &v);

    if (v == 1) {
      r += abs((i % 5) - 2); // # move row
      r += abs((i / 5) - 2); // # move column
    }
  }

  printf("%d", r);

  return 0;
}
