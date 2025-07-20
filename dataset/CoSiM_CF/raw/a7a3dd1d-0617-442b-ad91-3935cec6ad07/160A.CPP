#include <algorithm>
#include <iostream>
using namespace std;
main() {
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  reverse(arr, arr + n);
  int x = 0;
  for (int i = 0; i < n; i++) {
    x += arr[i];
    sum -= arr[i];
    if (x > sum) {
      cout << i + 1 << endl;
      return 0;
    }
  }
}
