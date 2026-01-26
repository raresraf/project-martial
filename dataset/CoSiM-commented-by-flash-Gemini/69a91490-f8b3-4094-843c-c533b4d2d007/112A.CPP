#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  string a, b;
  int i, r = 0;
  cin >> a;
  cin >> b;
  for (i = 0; a[i] != '\0'; i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    if (a[i] > b[i]) {
      r = 1;
      break;
    } else if (b[i] > a[i]) {
      r = -1;
      break;
    }
  }
  cout << r;
  return 0;
}
