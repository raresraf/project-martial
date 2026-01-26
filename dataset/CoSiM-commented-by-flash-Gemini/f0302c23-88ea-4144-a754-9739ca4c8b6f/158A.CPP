#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, k, count = 0;
  cin >> n >> k;
  int arr[n];
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  for (int j = 1; j <= n; j++) {
    if (arr[j] >= arr[k] && arr[j] > 0) {
      count++;
    }
  }
  cout << count;
}