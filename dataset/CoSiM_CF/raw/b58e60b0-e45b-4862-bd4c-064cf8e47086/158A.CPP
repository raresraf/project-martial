#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int arr[n], count = 0, max;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    if (i + 1 == k) {
      max = arr[i];
    }
  }
  for (int i = 0; i < n; i++) {
    if (arr[i] >= max && arr[i] > 0) {
      count++;
    }
  }
  cout << count << endl;
  return 0;
}