#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;

int main() {
  char k[100000], m[10000000];
  int i = 1, t, j;
  scanf("%d", &t);
  while (t--) {
    gets(k);
    strcpy(k, m);
    printf("OK\n");

    if (strcmp(m, k) == 0)
      cout << k << i << endl;
    i++;
  }
  return 0;
}
