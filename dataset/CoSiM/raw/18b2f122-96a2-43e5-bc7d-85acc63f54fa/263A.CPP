// Accepted
// Ahsanul kabir
#include <cstdlib>
#include <iostream>
using namespace std;
int main() {
  int a[10][10], row, col;
  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }
  int out = abs(3 - row) + abs(3 - col);
  cout << out << endl;
}
