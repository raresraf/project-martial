#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int arr[10][10];
  int i, j, a, b;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        a = i;
        b = j;
      }
    }
  }

  int ans = abs(3 - a) + abs(3 - b);
  cout << ans << endl;
  return 0;
}