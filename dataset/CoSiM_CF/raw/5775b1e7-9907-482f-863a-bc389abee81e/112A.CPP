#include <bits/stdc++.h>
using namespace std;
int main(void) {
  char a[102], b[102];
  int i;
  cin >> a;
  cin >> b;
  for (i = 0; a[i] != '\0'; i++) {
    if (isupper(a[i]))
      a[i] = tolower(a[i]);
    if (isupper(b[i]))
      b[i] = tolower(b[i]);
  }
  cout << strcmp(a, b);
}
