#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int arr[n + 1];
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  int reqScore = arr[k], count = 0;
  for (int i = 1; i <= n; i++) {
    if ((arr[i] > 0) && (arr[i] >= reqScore)) {
      count += 1;
    }
  }
  cout << count << "\n";
  return 0;
}
