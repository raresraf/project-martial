#include <iostream>
#include <string>
using namespace std;

int main() {
  int counter = 0;
  int row = 2;
  int coloum = 2;
  string matrix[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      char temp;
      cin >> temp;
      matrix[i][j] = temp;
      if (temp == '1') {
        row = i;
        coloum = j;
      }
    }
  }
  while (row != 2) {
    if (row < 2) {
      for (int i = 0; i < 5; i++) {
        string temp = matrix[row][i];
        matrix[row][i] = matrix[row + 1][i];
        matrix[row + 1][i] = temp;
      }
      row++;
    } else if (row > 2) {
      for (int i = 0; i < 5; i++) {
        string temp = matrix[row][i];
        matrix[row][i] = matrix[row - 1][i];
        matrix[row - 1][i] = temp;
      }
      row--;
    }
    counter++;
  }
  while (coloum != 2) {
    if (coloum < 2) {
      for (int i = 0; i < 5; i++) {
        string temp = matrix[i][coloum];
        matrix[i][coloum] = matrix[i][coloum + 1];
        matrix[i][coloum + 1] = temp;
      }
      coloum++;
    } else if (coloum > 2) {
      for (int i = 0; i < 5; i++) {
        string temp = matrix[i][coloum];
        matrix[i][coloum] = matrix[i][coloum - 1];
        matrix[i][coloum - 1] = temp;
      }
      coloum--;
    }
    counter++;
  }
  cout << counter;
  return 0;
}
