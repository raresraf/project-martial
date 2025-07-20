#include <stdio.h>

int main() {
  short n, t;
  scanf("%hd %hd", &n, &t);
  char arr[n + 1];
  scanf("%s", arr);
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (arr[i] == 'B' && arr[i + 1] == 'G') {
        char aux = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = aux;
        i++;
      }
    }
  }
  printf("%s", arr);

  return 0;
}
