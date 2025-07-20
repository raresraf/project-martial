#include <iostream>
using namespace std;

int main() {
  int t, n, maxb = 0;
  cin >> t;
  long long ans = 0;
  while (t-- > 0) {
    cin >> n;
    int arr[n];
    cin >> arr[0];
    maxb = arr[0];
    ans = 0;
    for (int i = 1; i < n; i++) {
      cin >> arr[i];
      if ((arr[i] > 0 && arr[i - 1] < 0) || (arr[i] < 0 && arr[i - 1] > 0)) {
        ans += maxb;
        maxb = arr[i];
      } else
        maxb = max(arr[i], maxb);
    }
    ans += maxb;
    maxb = 0;
    if (n == 1)
      ans = arr[0];
    cout << ans << endl;
  }
  return 0;
}
