#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  string a;
  int i, j, n;
  cin >> a;
  n = a.size();
  for (i = 0; i < n; i++) {
    a[i] = tolower(a[i]);
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y')
      continue;
    else
      cout << "." << a[i];
  }
  return 0;
}
