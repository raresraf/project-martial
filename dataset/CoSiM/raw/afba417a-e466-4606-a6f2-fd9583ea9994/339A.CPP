#include <bits/stdc++.h>
using namespace std;
int main() {
  long long n, m, time = 0;
  cin >> n >> m;
  long long arr[m + 1];
  for (int i = 0; i < m; i++)
    cin >> arr[i];
  time = arr[0] - 1;
  for (int i = 0; i < m - 1; i++) {
    if (arr[i + 1] < arr[i])
      time += (n - arr[i] + 1) + (arr[i + 1] - 1);
    else if (arr[i + 1] > arr[i])
      time += arr[i + 1] - arr[i];
  }
  cout << time << endl;
}
