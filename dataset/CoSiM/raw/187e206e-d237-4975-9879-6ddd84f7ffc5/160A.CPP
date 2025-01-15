// 56621300	Jul/06/2019 18:02UTC+6	havoccboi	160A - Twins	GNU
// C++17	Accepted	62 ms	0 KB

#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  int arr[t];
  int s = 0;
  for (int i = 0; i < t; i++) {
    cin >> arr[i];
    s += arr[i];
  }
  sort(arr, arr + t);
  int avg = s / 2;
  avg++;
  s = 0;
  int c = 0;
  // int i = sizeof(arr)/sizeof(arr[0])-1;
  int i = t - 1;
  while (s < avg) {
    s += arr[i];
    c++;
    i--;
  }

  cout << c << endl;
}
