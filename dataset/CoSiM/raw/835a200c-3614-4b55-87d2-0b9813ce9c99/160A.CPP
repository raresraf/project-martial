#include <iostream>
using namespace std;
int main() {
  // your code goes here
  int n, i, temp, j, sum = 0;
  cin >> n;
  int a[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum = sum + a[i];
  }
  sum = sum / 2;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n - i - 1; j++) {
      if (a[j] > a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }
  j = 0;
  temp = 0;
  for (i = n - 1; i >= 0; i--) {
    if (temp > sum)
      break;
    else {
      temp = temp + a[i];
      j++;
    }
  }
  cout << j;
  return 0;
}