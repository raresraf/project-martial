#include <iostream>
using namespace std;

int main() {
  int n, s, i, j;
  char a[50];
  cin >> n >> s;
  for (i = 0; i < n; i++)
    cin >> a[i];
  for (i = 0; i < s; i++)
    for (j = 0; j < n - 1; j++)
      if (a[j] == 'B' && a[j + 1] == 'G') {
        a[j] = 'G';
        a[++j] = 'B';
      }
  for (i = 0; i < n; i++)
    cout << a[i];

  return 0;
}
