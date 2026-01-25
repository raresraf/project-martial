

#include <iostream>

using namespace std;

int main() {
  int arr[5][5], row, col;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }
  cout << abs(2 - row) + abs(2 - col);
  return 0;
}

// Write your code here
