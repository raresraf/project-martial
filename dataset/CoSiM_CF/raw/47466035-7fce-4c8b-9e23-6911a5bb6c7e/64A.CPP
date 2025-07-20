#include <bits/stdc++.h>
using namespace std;
int main() {
  int l = 0, sum = 0, sum2 = 0, n;
  cin >> n;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  for (int i = 0; i < n; i++) {
    if ((arr[i] == 2 && arr[i + 1] == 3) || (arr[i] == 3 && arr[i + 1] == 2)) {
      l = 0;
      cout << "Infinite" << endl;
      break;
    } else {

      l = 1;
      sum2 = arr[i] + arr[i + 1];
      sum = sum + sum2;
      if (i == n - 2)
        break;
    }
  }
  if (l == 1) {
    cout << "Finite" << endl;
    cout << sum << endl;
  }
  return 0;
}
