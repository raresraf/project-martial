#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
  int n, co = 0;
  scanf("%d", &n);
  getchar();
  for (int i = 0; i < n; i++) {
    char arr[4];
    scanf("%s", &arr);
    if (arr[1] == '+') {
      co++;
    } else if (arr[1] == '-') {
      co--;
    }
  }
  printf("%d\n", co);
  return 0;
}
