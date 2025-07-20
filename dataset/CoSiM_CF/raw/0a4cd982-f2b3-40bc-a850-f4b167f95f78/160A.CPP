#include <algorithm>
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, arr[101], count1 = 0, sum1 = 0, sum2 = 0;
  cin >> n;
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  sort(arr, arr + n);
  int k = n - 1;
  while (1) {
    for (int i = n - 1; i >= k; i--) {
      sum1 += arr[i];
      count1++;
      // cout<<count1<<" ";
    }
    for (int j = 0; j < k; j++) {
      sum2 += arr[j];
    }
    if (sum1 > sum2) {
      break;
    } else {
      k--;
      count1 = 0;
      sum1 = 0;
      sum2 = 0;
    }
  }
  cout << count1 << endl;
}
