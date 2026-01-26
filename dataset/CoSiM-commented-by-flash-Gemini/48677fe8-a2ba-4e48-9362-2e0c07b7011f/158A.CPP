#include <iostream>

using namespace std;

main() {
  int n, k;
  cin >> n;
  cin >> k;
  int arr[n];
  int i, no = 0;
  for (i = 0; i < n; i++)
    cin >> arr[i];
  int x = arr[k - 1];
  for (i = 0; i < n; i++) {
    if (arr[i] >= x && arr[i] > 0)
      no++;
  }
  cout << no << endl;
}
