#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  int money = 0;
  int countt = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum = sum + arr[i];
  }
  sort(arr, arr + n);
  reverse(arr, arr + n);
  for (int i = 0; i < n; i++) {
    money = money + arr[i];
    sum = sum - arr[i];
    countt++;
    if (money > sum) {
      break;
    }
  }
  cout << countt;
  return 0;
}
