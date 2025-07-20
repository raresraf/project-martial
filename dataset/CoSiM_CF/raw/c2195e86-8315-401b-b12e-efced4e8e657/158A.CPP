#include <iostream>
using namespace std;
int main() {
  int arr[100], n, k, c = 0;
  while (cin >> n >> k) {
    for (int i = 0; i < n; i++) {
      cin >> arr[i];
    }
    for (int i = 0; i < n; i++) {
      if (arr[i] >= arr[k - 1] && arr[i] > 0)
        c++;
    }
    cout << c << endl;
    c = 0;
  }
}