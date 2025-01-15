#include <bits/stdc++.h>
using namespace std;

char c[55];

int main() {
  int n, cn = 0;
  scanf("%d", &n);
  cin.ignore();
  int i;
  for (i = 1; i <= n; i++) {
    scanf("%c", &c[i]);
  }
  int j;
  for (j = 1; j <= n; j++) {
    if (c[j] == c[j + 1])
      cn++;
  }
  printf("%d\n", cn);
  return 0;
}
