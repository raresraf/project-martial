//
// Created by Rivarrl on 2020/5/28.
//
/**
 * #96-A. Football
 *         7
 */
#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

int main() {
  char a[105];
  int p;
  while (scanf("%s", a) != EOF) {
    p = strlen(a);
    int cur = 1;
    for (int i = 1; i < p; i++) {
      if (a[i] == a[i - 1])
        cur++;
      else
        cur = 1;
      if (cur == 7) {
        printf("YES\n");
        break;
      }
    }
    if (cur < 7)
      printf("NO\n");
    memset(a, 0, sizeof(a));
  }
  return 0;
}