#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int n, i;
  char a[101];
  cin >> a;
  for (i = 0; i < strlen(a); i++) {
    if (!(a[i] == 'A' || a[i] == 'E' || a[i] == 'I' || a[i] == 'O' ||
          a[i] == 'U' || a[i] == 'a' || a[i] == 'e' || a[i] == 'i' ||
          a[i] == 'o' || a[i] == 'u' || a[i] == 'y' || a[i] == 'Y')) {
      if (a[i] >= 65 && a[i] <= 90) {
        a[i] += 32;
      }
      cout << "." << a[i];
    }
  }
}
