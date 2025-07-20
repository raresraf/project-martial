/* P160A by Gabriel Scotillo */
#include <bits/stdc++.h>

using namespace std;

void swap(int a[], int j, int i) {
  int k = a[j];
  a[j] = a[i];
  a[i] = k;
}

void ordenaDecreciente(int a[], int N) {
  int i = 0;
  while (i < N) {
    int j = i + 1;
    while (j < N) {
      if (a[j] >= a[i]) {
        swap(a, j, i);
      }
      j++;
    }
    i++;
  }
}

int main() {
  int res = 0;
  int n;
  cin >> n;
  const int N = n;
  int sumTot = 0;
  int sumParcial = 0;
  int a[N];
  for (int i = 0; i < N; i++) {
    cin >> a[i];
    sumTot = sumTot + a[i];
  }

  ordenaDecreciente(a, N);

  for (int i = 0; i < N; i++) {
    if (sumParcial > sumTot - sumParcial) {
      break;
    } else {
      sumParcial += a[i], res++;
    }
  }

  cout << res << endl;
  return 0;
}
