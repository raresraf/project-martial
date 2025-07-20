#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  int xpos, ypos;
  int moves = 0;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        xpos = i;
        ypos = j;
      }
    }
  }
  moves += abs(xpos - 2) + abs(ypos - 2);
  cout << moves << '\n';
  return 0;
}