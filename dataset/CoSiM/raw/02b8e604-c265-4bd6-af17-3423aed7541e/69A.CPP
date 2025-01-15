#include <iostream>
using namespace std;

int main() {
  int n;
  int x = 0, y = 0, z = 0;
  cin >> n;
  int arr[3 * n];
  for (int i = 0; i < 3 * n; i++) {
    cin >> arr[i];
  }
  x = arr[0] + arr[3] + arr[6];
  y = arr[1] + arr[4] + arr[7];
  z = arr[2] + arr[5] + arr[8];

  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
