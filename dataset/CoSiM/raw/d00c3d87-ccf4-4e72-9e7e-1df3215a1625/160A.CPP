#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, s = 0, s1 = 0, c = 0;
  cin >> n;
  int arr[n];
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    s = s + arr[i];
  }
  sort(arr, arr + n);
  for (i = n - 1; i >= 0; i--) {
    c++;
    s1 = s1 + arr[i];
    if (s1 > s - s1) {
      cout << c;
      return 0;
    }
  }
}
