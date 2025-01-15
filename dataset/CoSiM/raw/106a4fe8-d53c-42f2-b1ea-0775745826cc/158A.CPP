#include <bits/stdc++.h>
using namespace std;
int main() {
  int a, b, p, c = 0, flag = 0;
  cin >> a >> b;
  int arr[a];
  for (int i = 0; i < a; i++) {
    cin >> arr[i];
    if (i == b - 1) {
      p = arr[i];
    }
  }
  // cout<<p<<endl;
  for (int j = 0; j < a; j++) {
    if (arr[j] >= p && arr[j] > 0) {
      c++;
    }
  }
  cout << c << endl;
}
