#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, temp;
  cin >> n;
  int arr[n];
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (arr[i] > arr[j]) {
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }

  int total = 0, k = 1, total1 = 0;
  for (int j = 0; j < n; j++) {
    total = 0;
    for (int i = 0; i < n - k; i++)
      total += arr[i];
    for (int i = n - k; i > (n - k - 1); i--)
      total1 += arr[i];
    if (total < total1) {
      cout << k << endl;
      break;
    }
    k++;
  }

  return 0;
}
