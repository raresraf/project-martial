#include <iostream>
using namespace std;
int main() {
  int i, c = 0;
  char a[102];
  cin >> a;
  for (i = 0; a[i] != '\0'; i++) {
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y' || a[i] == 'A' || a[i] == 'E' ||
        a[i] == 'I' || a[i] == 'O' || a[i] == 'U' || a[i] == 'Y')
      c = 1;
    if (c == 0) {
      if (a[i] >= 'A' && a[i] <= 'Z') {
        a[i] = a[i] + 32;
        cout << "." << a[i];
      } else
        cout << "." << a[i];
    }
    c = 0;
  }
  return 0;
}
