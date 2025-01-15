#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int arr[101], total = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    total += arr[i];
  }
  int coin = 0, cnt = 0, i = n - 1;
  sort(arr, arr + n);
  while (coin <= total && i >= 0) {
    coin += arr[i];
    total -= arr[i];
    cnt++;
    i--;
  }
  cout << cnt;
}