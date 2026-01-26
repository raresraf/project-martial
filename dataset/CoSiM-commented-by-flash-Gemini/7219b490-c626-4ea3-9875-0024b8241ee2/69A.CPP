#include <iostream>
using namespace std;

int main() {
  int n, i, j;
  cin >> n;
  int arr[n][3];
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }
  bool ans = true;
  for (j = 0; j < 3; j++) {
    int sum = 0;
    for (i = 0; i < n; i++) {
      sum += arr[i][j];
    }
    if (sum != 0) {
      ans = false;
    }
  }
  if (ans) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
