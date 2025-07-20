#include <iostream>

using namespace std;
int main() {
  int i, n, k, ans, array[100];
  cin >> n >> k;
  for (i = 0; i < n; i++) {
    cin >> array[i];
  }
  // case 1 : array[k-1] > 0
  if (array[k - 1] > 0) {
    // either array[k] != array[k-1]
    if (array[k] != array[k - 1])
      ans = k;
    // or else
    else {
      ans = k;
      for (i = k; i < n; i++) {
        if (array[i] == array[i - 1])
          ans++;
        else
          break;
      }
    }
  }
  // case 2 : array[k-1] == 0
  else {
    ans = k - 1;
    for (i = k - 2; i >= 0; i--) {
      if (array[i] == 0)
        ans--;
      else
        break;
    }
  }
  cout << ans << endl;
  return 0;
}
