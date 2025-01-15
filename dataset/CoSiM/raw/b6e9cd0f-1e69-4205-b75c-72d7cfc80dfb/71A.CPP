#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    char ar[101];
    scanf("%s", ar);
    int co = 0;
    for (int j = 0; ar[j] != '\0'; j++) {
      co++;
    }
    if (co < 11) {
      printf("%s\n", ar);
    } else {
      printf("%c%d%c\n", ar[0], co - 2, ar[co - 1]);
    }
  }
  return 0;
}
