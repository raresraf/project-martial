#include <iostream>
using namespace std;
char cha(char a[], int n) {
  char b[n];
  for (int i = 0; i < n; i++)
    b[i] = a[i];
  for (int i = 0; i < n - 1; i++) {
    if (b[i] == 'B' && b[i + 1] == 'G') {
      a[i] = 'G';
      a[i + 1] = 'B';
    }
  }
  return *a;
}
int main() {
  int n, t;
  cin >> n >> t;
  char a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  for (int i = 0; i < t; i++)
    *a = cha(a, n);
  for (int i = 0; i < n; i++)
    cout << a[i];
}
