#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;

#define sz 101
int ara[sz + 3];

int main() {

  int i, cont, k, n, loc;

  while (scanf("%d%d", &n, &k) == 2) {

    cont = 0;
    for (i = 1; i <= n; i++) {
      scanf("%d", &ara[i]);
      if (i == k)
        loc = ara[i];
    }

    for (i = 1; i <= n; i++) {
      if (ara[i] >= loc && ara[i] > 0) {
        cont++;
      }
    }

    printf("%d\n", cont);
  }

  return 0;
}
