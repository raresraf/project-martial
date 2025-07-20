#include <iostream>

using namespace std;

int main() {

  int n;
  int ans = 0;

  cin >> n;

  int a[n] = {0};
  int b[n] = {0};
  int tmp[n] = {0};

  for (int i = 1; i <= n; ++i) {
    cin >> a[i] >> b[i];
  }

  tmp[0] = b[1];

  for (int i = 2; i <= n; ++i) {

    tmp[i - 1] = b[i - 1] - a[i];
    tmp[i - 1] += b[i];

    b[i] = tmp[i - 1];
  }

  for (int i = 0; i < n; ++i) {
    if (tmp[i] > ans) {
      ans = tmp[i];
    }
    // cout<<tmp[i];
  }

  cout << ans;

  return 0;
}
