#include <iostream>
using namespace std;

int abs(int i) {
  if (i < 0) {
    return -i;
  }
  return i;
}

int main() {
  int a[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        cout << abs(i - 2) + abs(j - 2);
        break;
      }
    }
  }
}
