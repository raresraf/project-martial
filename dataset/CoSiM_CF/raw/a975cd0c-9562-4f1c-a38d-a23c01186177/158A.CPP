#include <iostream>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int arr[n - 1];
  for (int i = 0; i < k; i++) {
    cin >> arr[i];
    if (arr[i] == 0) {
      cout << i << endl; // 10 9 8 7 7 7 5 5
      return 0;
    }
  }
  for (int i = k; i < n; i++) {
    cin >> arr[i];
    if (arr[i] == arr[i - 1]) {
      k++;
    } else {
      break;
    }
  }
  cout << k << endl;
}
