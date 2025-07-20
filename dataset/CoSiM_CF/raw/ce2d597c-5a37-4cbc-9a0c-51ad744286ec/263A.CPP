#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int arr[5][5];
  int iindex, jindex;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        iindex = i;
        jindex = j;
      }
    }
  }
  cout << abs(2 - iindex) + abs(2 - jindex) << endl;

  return 0;
}
