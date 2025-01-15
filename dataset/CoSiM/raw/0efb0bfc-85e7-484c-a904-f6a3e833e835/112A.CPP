#include <ctype.h>
#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int i, j, s;
  char a[101], b[101];
  cin >> a >> b;
  for (i = 0; a[i] != '\0'; i++) {
    if (a[i] >= 'A' && a[i] <= 'Z')
      a[i] = a[i] + 32;
    if (b[i] >= 'A' && b[i] <= 'Z')
      b[i] = b[i] + 32;
  }
  s = strcmp(a, b);
  if (s == 0)
    cout << "0";
  if (s > 0)
    cout << "1";
  if (s < 0)
    cout << "-1";
  return 0;
}
