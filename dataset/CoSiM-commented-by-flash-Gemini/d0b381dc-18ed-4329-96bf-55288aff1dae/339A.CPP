#include <bits/stdc++.h>
using namespace std;
int main() {
  char arr[101];
  scanf("%s", &arr);
  int len = strlen(arr);
  int x, temp, i, j;
  x = (len + 1) / 2;
  int a[x];
  for (i = 0, j = 0; i < len; i += 2, j++) {
    if (arr[i] == '1') {
      a[j] = 1;
    } else if (arr[i] == '2') {
      a[j] = 2;
    } else if (arr[i] == '3') {
      a[j] = 3;
    }
  }
  for (i = 0; i < x - 1; i++) {
    for (j = 0; j < x - 1 - i; j++) {
      if (a[j] > a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }
  for (i = 0, j = 0; i < len; i += 2, j++) {
    printf("%d", a[j]);
    if (i == len - 1) {
      break;
    }
    printf("%c", '+');
  }
  return 0;
}
