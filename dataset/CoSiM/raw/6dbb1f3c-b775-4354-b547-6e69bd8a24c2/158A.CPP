#include <iostream>

int main() {
  using namespace std;

  int n, k;
  int arr[50];
  int i;
  int pos;
  int count = 0;

  cin >> n >> k;

  for (i = 0; i < n; i++) {
    cin >> arr[i];
  }

  for (i = 0; i < n; i++) {
    if ((arr[i] >= arr[k - 1]) && (arr[i] > 0))

      count++;
  }

  cout << count;
  return 0;
}
