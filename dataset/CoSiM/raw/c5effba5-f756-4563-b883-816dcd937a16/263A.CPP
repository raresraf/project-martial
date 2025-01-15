#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }
  int m = 0, n = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] == 1) {
        m = i;
        n = j;
      }
    }
  }
  m = 2 - m;
  n = 2 - n;
  if (m < 0) {
    m = (-1) * m;
  }
  if (n < 0) {
    n = (-1) * n;
  }
  cout << m + n << endl;
  return 0;
}
