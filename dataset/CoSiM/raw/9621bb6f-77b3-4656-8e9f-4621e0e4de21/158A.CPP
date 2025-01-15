#include <iostream>

using namespace std;

int main() {
  int n, k, count = 0, arr[100];

  cin >> n >> k;

  for (int i = 0; i < n; i++)
    cin >> arr[i];

  for (int i = 0; i < n; i++) {
    if ((arr[i] >= arr[k - 1]) && arr[i] > 0)
      count++;
  }

  cout << count;

  return 0;
}