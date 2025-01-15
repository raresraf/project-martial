#include <algorithm>
#include <iostream>
using namespace std;
int a[100];

int main() {
  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  sort(a, a + n, greater<int>());

  int num = 0;
  int YourSum = 0;

  for (int i = 0; i < n; i++) {
    num++;
    YourSum += a[i];
    int OtherSum = 0;
    for (int j = i + 1; j < n; j++) {
      OtherSum += a[j];
    }

    if (YourSum > OtherSum) {
      break;
    }
  }

  cout << num;

  return 0;
}