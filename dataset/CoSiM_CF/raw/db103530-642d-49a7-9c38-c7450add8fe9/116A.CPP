#include <iostream>
using namespace std;
int n, a, b;

int main() {
  cin >> n;
  int arr[n];
  int init = 0;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    init = init - a + b;
    arr[i] = init;
  }

  int m = 0;

  for (int i = 0; i < n; i++) {
    if (arr[i] > m)
      m = arr[i];
  }

  cout << m << endl;

  return 0;
}
