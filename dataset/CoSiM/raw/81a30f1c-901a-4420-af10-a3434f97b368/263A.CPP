#include <iostream>
#include <math.h>
using namespace std;
void solved() {
  int a;
  int row, column;
  bool check = false;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a;
      if (a == 1) {
        row = i;
        column = j;
        check = true;
        break;
      }
    }
    if (check)
      break;
  }
  int count = 0;
  count += abs(row - 2);
  count += abs(column - 2);
  cout << count;
}
int main() {
  solved();
  return 0;
}
