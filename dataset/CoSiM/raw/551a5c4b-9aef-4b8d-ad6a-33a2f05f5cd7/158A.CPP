#include <iostream>
using namespace std;
int main() {
  int n, k, i, a;
  cin >> n >> k;
  int arr[n];
  a = 0;
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    if (arr[i] == 0 && i == 0) {
      cout << 0;
      return 0;
    }
  }
  for (i = 0; i < n; i++) {
    if (arr[i] < arr[k - 1])
      break;
    else if (arr[i] >= arr[k - 1] && arr[i] != 0)
      a++;
  }
  cout << a;
  return 0;
}
