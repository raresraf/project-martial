#include <iostream>
using namespace std;
int main() {
  int a, c, i, j, d = 0, e = 0, f = 0;
  cin >> a;
  int b[1000];
  for (i = 0; i < a; i++) {
    cin >> b[i];
    d += b[i];
  }
  for (i = 0; i < (a - 1); i++) {
    for (j = 0; j < a - i - 1; j++) {
      if (b[j] < b[j + 1]) {
        c = b[j];
        b[j] = b[j + 1];
        b[j + 1] = c;
      }
    }
  }
  for (i = 0; i < a; i++) {

    e += b[i];
    f++;
    if (e > (d / 2))
      break;
  }

  cout << f;
}
