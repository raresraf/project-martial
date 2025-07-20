//
// Created by Taro Young on 2020/9/6.
//
#include <iostream>

using namespace std;

int main() {
  int n, k;
  int a[110];
  cin >> n >> k;
  for (int i = 0; i < n; i++)
    cin >> a[i];
  if (a[k - 1] > 0)
    while (a[k] == a[k - 1])
      k++;
  else
    while (a[k - 1] <= 0 && k > 0)
      k--;
  cout << k << endl;
  return 0;
}
