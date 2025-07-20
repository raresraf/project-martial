#include <bits/stdc++.h>
using namespace std;
int main() {
  int arr[105];
  int n, i, cnt = 0, k, cnti = 0, cntt = 1;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  sort(arr, arr + n);
  k = n - 1;
  while (cnti <= cntt) {
    cnti = 0;
    cntt = 0;
    for (int i = 0; i < k; i++) {
      cntt += arr[i];
    }
    for (int j = n - 1; j >= k; j--) {
      cnti += arr[j];
    }
    cnt++;
    k--;
  }
  cout << cnt;
  return 0;
  //    i=0;
  //    while(n--)
  //    {
  //        cout<<arr[i++]<<" ";
  //    }
}
