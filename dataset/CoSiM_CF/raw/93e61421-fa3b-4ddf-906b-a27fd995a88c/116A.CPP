#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  long long t;
  long long maxx = 0;
  cin >> t;
  long long arr[2 * t];
  for (int i = 0; i < 2 * t; i++) {
    cin >> arr[i];
  }

  long long sum = 0;

  for (int i = 0; i < 2 * t; i++) {
    if (i % 2 == 0) {
      sum = sum - arr[i];
      if (sum > maxx) {
        maxx = sum;
      }
    } else if (i % 2 != 0) {
      sum = sum + arr[i];
      if (sum > maxx) {
        maxx = sum;
      }
    }
  }
  cout << maxx;
  return 0;
}
