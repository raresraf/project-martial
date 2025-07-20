#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int cheater = 0, sum = 0;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  int i;
  for (i = n - 1; i >= 0; i--) {
    cheater += arr[i];
    if (cheater * 2 > sum)
      break;
  }
  cout << n - i << endl;
  return 0;
}
