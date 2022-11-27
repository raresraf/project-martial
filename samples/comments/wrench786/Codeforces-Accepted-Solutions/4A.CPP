#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n;
  scanf("%d", &n);
  if (n == 2 || n % 2 == 1) {
    printf("NO\n");
  } else if (n % 2 == 0) {
    printf("YES\n");
  }
  return 0;
}
