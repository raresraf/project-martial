#include <bits/stdc++.h>
using namespace std;

int main() {
  int N;
  cin >> N;

  int arr[N][3];

  int x = 0, y = 0, z = 0;

  for (int i = 0; i < N; i++) {
    cin >> arr[i][0] >> arr[i][1] >> arr[i][2];
    x += arr[i][0];
    y += arr[i][1];
    z += arr[i][2];
  }

  if (x == 0 and y == 0 and z == 0)
    cout << "YES";

  else
    cout << "NO";
}
