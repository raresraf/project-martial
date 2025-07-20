#include <iostream>

using namespace std;

int main() {
  int input;
  int rowCount, colCount;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> input;
      if (input == 1) {
        rowCount = i;
        colCount = j;
      }
    }
  }

  cout << abs(rowCount - 2) + abs(colCount - 2) << endl;
  return 0;
}