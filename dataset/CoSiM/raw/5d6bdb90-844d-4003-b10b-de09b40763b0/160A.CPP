#include <algorithm>
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  int count = 0, sum2 = 0;
  for (int i = n - 1; i >= 0; i--) {
    if (sum2 > sum / 2) {
      break;
    } else {
      sum2 += arr[i];
      count++;
    }
  }
  cout << count;
  return 0;
}
