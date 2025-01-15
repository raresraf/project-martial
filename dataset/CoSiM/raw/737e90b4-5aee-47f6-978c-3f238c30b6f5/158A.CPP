#include <iostream>

using namespace std;

int main() {
  int n, k, i, s = 0;
  cin >> n;
  cin >> k;
  int arr[n];
  for (i = 0; i < n; i++) {
    cin >> arr[i];
  }
  for (i = 0; i < n; i++) {
    if ((arr[i] >= arr[k - 1]) && (arr[i] > 0)) {
      s++;
    }
  }
  cout << s;
  return 0;
}
