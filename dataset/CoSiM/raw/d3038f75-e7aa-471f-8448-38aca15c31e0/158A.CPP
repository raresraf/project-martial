#include <iostream>
using namespace std;
int main() {
  int n, k, Total = 0;
  cin >> n >> k;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  k = arr[k - 1];
  for (int i = 0; i < n; i++) {
    if (arr[i] >= k && arr[i] > 0) {
      Total++;
    }
  }
  cout << Total;
  return 0;
}
