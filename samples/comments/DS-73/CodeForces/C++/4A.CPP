#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k, temp;
  cin >> n >> k;

  int arr[50];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  int res = 0;
  for (int i = 0; i < n; i++) {
    if (arr[i] < 1) {
      break;
    } else if (arr[i] >= arr[k - 1]) {
      res++;
    }
  }
  cout << res;

  return 0;
}