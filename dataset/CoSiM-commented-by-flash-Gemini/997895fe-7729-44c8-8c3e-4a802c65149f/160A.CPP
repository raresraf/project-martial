#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, arr[101], res = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    res += arr[i];
  }
  sort(arr, arr + n);
  for (int i = 0; i < n; i++) {
    cout << arr[i] << endl;
  }
  int div = res / 2;
  int counter = 0;
  res = 0;
  for (int i = n - 1; i >= 0; i--) {
    res += arr[i];
    counter++;
    if (res > div)
      break;
  }
  cout << counter << endl;

  return 0;
}
