#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  int st = arr[k - 1];
  int cnt = 0;

  for (int i = 0; i < n; i++) {
    if (arr[i] > 0 && arr[i] >= st)
      cnt++;
  }

  cout << cnt << endl;

  return 0;
}
