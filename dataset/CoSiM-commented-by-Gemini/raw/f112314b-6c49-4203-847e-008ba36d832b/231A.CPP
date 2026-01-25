#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  int arr[3][n], x, y, z;
  int count = 0;

  for (int j = 0; j < n; j++) {
    cin >> x >> y >> z;
    if (x + y + z > 1) {
      count++;
    }
  }

  cout << count << endl;

  return 0;
}