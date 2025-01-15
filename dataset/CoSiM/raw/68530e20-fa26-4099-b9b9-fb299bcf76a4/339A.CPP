#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  int a[101], j = 0, length = 0, i, n;
  cin >> str;
  n = str.length() - 1;
  for (i = 0; i <= n; i++) {
    if (str[i] == '1' || str[i] == '2' || str[i] == '3') {
      int c = atoi(&str[i]);
      a[j] = c;
      j++;
      length++;
    }
  }
  sort(a, a + length);
  for (j = 0; j <= length - 1; j++) {
    printf("%d", a[j]);
    if (j < length - 1) {
      printf("+");
    }
  }
  printf("\n");
}
