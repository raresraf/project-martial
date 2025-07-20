#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  int l, f;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        l = i + 1;
        f = j + 1;
      }
    }
  }
  cout << abs(l - 3) + abs(f - 3);
}