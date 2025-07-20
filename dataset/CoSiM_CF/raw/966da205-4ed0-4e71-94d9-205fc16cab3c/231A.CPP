#include <bits/stdc++.h>
using namespace std;
int main(int argc, char const *argv[]) {
  /* code */
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int n;
  cin >> n;
  int arr[n][3];
  int ans = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  for (int i = 0; i < n; i++) {
    int k = 0;
    for (int j = 0; j < 3; j++) {
      if (arr[i][j] == 1) {
        k += 1;
      }
    }

    if (k >= 2) {
      ans++;
    }
  }
  cout << ans;

  return 0;
}