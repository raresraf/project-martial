#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int i, n, j;
  char a[101], b[101];
  cin >> a;
  n = strlen(a);
  for (i = 0, j = 0; i < n; i++) {
    if (a[i] <= 122 && a[i] >= 97) {
      if (a[i] != 97 && a[i] != 101 && a[i] != 105 && a[i] != 111 &&
          a[i] != 117 && a[i] != 121) {
        b[j] = a[i];
        j++;
      }
    } else {
      if (a[i] != 65 && a[i] != 69 && a[i] != 73 && a[i] != 79 && a[i] != 85 &&
          a[i] != 89) {
        b[j] = a[i] + 32;
        j++;
      }
    }
  }
  b[j] = '\0';
  for (i = 0; i < 2 * j; i++) {
    if (i % 2 == 0)
      cout << ".";
    else
      cout << b[i / 2];
  }
  return 0;
}
