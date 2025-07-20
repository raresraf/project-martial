#include <iostream>

using namespace std;

void swap(int &x, int &y) {
  int tmp;
  tmp = x;
  x = y;
  y = tmp;
}

int main() {
  int n, a[100] = {}, sum = 0, val = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  for (int i = 0; i < n - 1; i++) // bubble sort
  {
    for (int j = 0; j < n - 1; j++) {
      if (a[j] < a[j + 1])
        swap(a[j], a[j + 1]);
    }
  }
  int k = 0;
  while (val <= sum - val) {
    val += a[k];
    k++;
  }
  cout << k;
  return 0;
}