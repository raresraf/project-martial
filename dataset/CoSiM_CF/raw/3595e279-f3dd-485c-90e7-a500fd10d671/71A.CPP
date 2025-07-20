#include <iostream>
#include <string.h>
using namespace std;
int main() {
  char s[99];
  int n, l;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s;
    l = strlen(s);
    if (l > 10) {
      cout << s[0] << l - 2 << s[l - 1];
    } else {
      cout << s;
    }
  }
  return 0;
}