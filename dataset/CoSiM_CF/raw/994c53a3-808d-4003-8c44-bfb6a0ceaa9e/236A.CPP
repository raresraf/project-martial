#include "stdio.h"

int main() {
  bool rep[256] = {0};

  char c;
  int count = 0;
  while (scanf("%c", &c) && c != '\n' && c != '\0') {
    if (!rep[c]) {
      count++;
    }

    rep[c] = true;
  }

  printf(count % 2 ? "IGNORE HIM!" : "CHAT WITH HER!");
  return 0;
}
