#include <iostream>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int arr[n];
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  int count = 0;
  if (arr[0] > 0) {
    int prev = arr[0];

    int i = 0;
    int pos = 1;
    while (i < n) {
      if (arr[i] <= 0)
        break;
      if (arr[i] < prev) {
        prev = arr[i];
        pos = i + 1;
        if (pos > k)
          break;
      }
      count++;
      i++;
    }
  }
  cout << count;
  return 0;
}