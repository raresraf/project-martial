#include <bits/stdc++.h>
using namespace std;
int main() {
  char str[105];
  int a[105];
  int i, j;
  while (scanf("%s", str) != EOF) {
    for (i = 0, j = 0; i < strlen(str); i++) {
      if (str[i] >= '0' && str[i] <= '9') {
        a[j++] = str[i] - '0';
      }
    }
    sort(a, a + j);
    for (i = 0; i < j; i++) {
      cout << a[i];
      if (i < j - 1)
        cout << "+";
    }
    cout << endl;
  }
  return 0;
}
