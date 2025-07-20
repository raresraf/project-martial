

#include <iostream>

using namespace std;

#define f(i, n) for (int i = 0; i < n; i++)
void sort(int a[], int n) {
  int temp = 1;
  a[n] = 0;
  while (temp != 0) {
    temp = 0;
    f(i, n) {
      if (a[i] < a[i + 1]) {
        temp = a[i];
        a[i] = a[i + 1];
        a[i + 1] = temp;
      }
    }
  }
}

int coins(int a[], int n) {
  int min = 1, sum, total = 0;

  sum = a[0];

  for (int i = 0; i < n; i++) {

    total += a[i];
  }
  while (sum <= total / 2) {
    sum += a[min];
    ++min;
  }

  return min;
}

int main() {
  int n;

  cin >> n;
  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  sort(a, n);

  cout << coins(a, n);

  return 0;
}