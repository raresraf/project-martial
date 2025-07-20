#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int c = 0;
  for (int i = 0; i < n; i++) {
    int arr[3];
    int k = 0;
    cin >> arr[0];
    if (arr[0] == 1)
      k++;
    cin >> arr[1];
    if (arr[1] == 1)
      k++;
    cin >> arr[2];
    if (arr[2] == 1)
      k++;
    if (k > 1)
      c++;
  }
  cout << c << endl;

  return 0;
}
