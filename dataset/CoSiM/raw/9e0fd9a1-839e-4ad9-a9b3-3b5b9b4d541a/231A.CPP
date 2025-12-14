#include <iostream>
using namespace std;
int main() {
  int arr[3], i, j, n, flag = 0;
  cin >> n;
  if (1 <= n && n <= 1000) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < 3; j++) {
        cin >> arr[j];
      }
      if (arr[0] + arr[1] + arr[2] >= 2)
        flag++;
    }
    cout << flag << endl;
  }
  return 0;
}
