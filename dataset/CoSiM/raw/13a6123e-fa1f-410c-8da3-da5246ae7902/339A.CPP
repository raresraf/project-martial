#include <algorithm>
#include <iostream>
#include <string.h>
using namespace std;
int main() {

  char ch[102];
  int a[53], len;
  // freopen("339A.txt","r",stdin);
  cin >> ch;
  for (int i = 0, j = 0; ch[i] != '\0'; i += 2) {
    a[j++] = ch[i] - 48;
  }
  len = strlen(ch);
  sort(a, a + (len / 2) + 1);
  for (int i = 0, j = 0; i < (len / 2) + 1; i++) {
    if (i != (len / 2))
      cout << a[i] << "+";
    else
      cout << a[i];
  }

  return 0;
}
