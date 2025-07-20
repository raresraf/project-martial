#include <cstdlib>
#include <iostream>
using namespace std;

int main() {
  int arr[5][5], i, j, l, k;

  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        l = i;
        k = j;
      }
    }
  }
  l = abs(2 - l);
  k = abs(2 - k);

  cout << l + k;
  return 0;
}