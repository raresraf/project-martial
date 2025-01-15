#include <iostream>

using namespace std;

int main() {
  int n, i, j, s = 0, f = 0;
  int arr[3];
  cin >> n;
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> arr[j];
      if (arr[j] == 1) {
        s++;
      }
    }
    if (s >= 2) {
      f++;
    }
    s = 0;
  }
  cout << f;
  return 0;
}
