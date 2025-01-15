// solution to codeforces problem number 158-A , Problem title - Next Round ,
// file format cpp , compile through gcc ++17
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int arr[n];

  for (int i = 0; i < n; i++)
    cin >> arr[i];
  int c = 0;
  if (arr[k - 1] > 0) {
    for (int z : arr)
      if (z >= arr[k - 1])
        c++;
  }

  else {
    for (int z : arr)
      if (z > arr[k - 1])
        c++;
  }
  cout << c << endl;
}
