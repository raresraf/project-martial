// 29.01.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  int n;
  int t;
  char temp;

  cin >> n >> t;
  char a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  if (n < t)
    t = n;
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n; j++) {
      if (a[j] == 'B' && a[j + 1] == 'G') {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
        j++;
      }
    }
  }
  for (int i = 0; i < n; i++)
    cout << a[i];
  return (0);
}