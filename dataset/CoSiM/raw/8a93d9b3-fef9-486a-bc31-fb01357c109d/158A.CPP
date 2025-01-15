#include <iostream>
using namespace std;

int main(void) {
  int n, k;
  int arr[50];
  int i;

  cin >> n;
  cin >> k;

  if (n >= k) {
    for (i = 0; i < n; i++)
      cin >> arr[i];

    for (i = 0; i < n; i++)
      if (arr[i] < arr[k - 1] || arr[i] <= 0)
        break;

    cout << i << endl;
  }
}
