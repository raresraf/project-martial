#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, arr[102], cnt = 0, temp;
  cin >> n >> k;
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  temp = arr[k - 1];

  for (int i = 0; i < n; i++) {
    if (arr[i] >= temp && arr[i] > 0)
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}
